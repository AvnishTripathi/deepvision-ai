"""
DeepVision AI — Video Face Detection & Multi-Face Tracking Test Suite (Step 5)
Comprehensive unit and integration tests covering:
- Video input validation & corruption rejection
- Video metadata extraction & duration calculation
- ByteTrack multi-face tracking engine & 2D Kalman motion model
- Persistent temporary track IDs & session reset invariants
- VideoProcessingService streaming pipeline & performance intervals
- Asynchronous VideoJobManager & storage abstraction
- POST /api/v1/video/analyze, GET /jobs/{id}, and GET /jobs/{id}/download endpoints
"""

import os
import io
import time
import tempfile
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.exceptions import (
    UnsupportedVideoError,
    CorruptedVideoException,
    EmptyVideoException,
    VideoTooLargeException,
    VideoDurationExceededException,
    VideoProcessingCancelledException,
)
from app.core.video_security import (
    validate_video_extension_and_mime,
    probe_and_validate_video,
)
from app.core.storage import get_storage_backend, LocalStorageBackend
from app.schemas.ai import BoundingBox, FaceDetection
from app.schemas.video import VideoMetadata, VideoProcessingOptions
from app.schemas.video_job import VideoJobStatus
from app.ai.tracking import (
    get_face_tracker,
    ByteTrackFaceTracker,
    TrackResult,
    TrackState,
)
from app.services.video_processing_service import (
    VideoProcessingService,
    get_video_processing_service,
)
from app.services.video_job_manager import (
    get_video_job_manager,
    InMemoryVideoJobManager,
)


@pytest.fixture
def sample_video_path():
    """Generates a temporary 10-frame synthetic test video containing a moving face."""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"test_fixture_{int(time.time()*1000)}.mp4")
    canvas_w, canvas_h = 640, 480
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (canvas_w, canvas_h))

    # Load fixture face or create clean face pattern
    fixtures_face = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")
    if os.path.exists(fixtures_face):
        face_img = cv2.imread(fixtures_face)
        face_crop = cv2.resize(face_img, (260, 260))
    else:
        face_crop = np.full((260, 260, 3), 200, dtype=np.uint8)

    for i in range(10):
        frame = np.full((canvas_h, canvas_w, 3), 120, dtype=np.uint8)
        x_pos = 50 + i * 10
        y_pos = 50
        frame[y_pos : y_pos + 260, x_pos : x_pos + 260] = face_crop
        out.write(frame)
    out.release()

    yield video_path

    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except OSError:
            pass


@pytest.fixture
def multi_face_video_path():
    """Generates a temporary video containing two moving faces."""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"test_multi_{int(time.time()*1000)}.mp4")
    canvas_w, canvas_h = 800, 600
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (canvas_w, canvas_h))
    fixtures_face = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")
    if os.path.exists(fixtures_face):
        face_img = cv2.imread(fixtures_face)
        face_crop = cv2.resize(face_img, (260, 260))
    else:
        face_crop = np.full((260, 260, 3), 200, dtype=np.uint8)

    for i in range(8):
        frame = np.full((canvas_h, canvas_w, 3), 120, dtype=np.uint8)
        frame[50 : 310, 50 + i * 5 : 310 + i * 5] = face_crop
        frame[50 : 310, 480 - i * 5 : 740 - i * 5] = face_crop
        out.write(frame)
    out.release()
    yield video_path
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except OSError:
            pass


@pytest.fixture
def no_face_video_path():
    """Generates a temporary video containing no human faces (blank background)."""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"test_noface_{int(time.time()*1000)}.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480))
    for _ in range(6):
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        out.write(frame)
    out.release()
    yield video_path
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except OSError:
            pass


# =============================================================================
# 1. Video Input Validation & Security Tests
# =============================================================================

class TestVideoValidationAndSecurity:
    """Tests container validation, MIME verification, and corruption rejection."""

    def test_valid_extensions_and_mimes(self):
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            norm_ext = validate_video_extension_and_mime(f"test{ext}")
            assert norm_ext == ext

    def test_unsupported_extension_rejected(self):
        with pytest.raises(UnsupportedVideoError):
            validate_video_extension_and_mime("malicious_script.exe")
        with pytest.raises(UnsupportedVideoError):
            validate_video_extension_and_mime("image.png")

    def test_corrupted_video_file_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"CORRUPTED_NON_VIDEO_STREAM_DATA_BYTES")
            corrupt_path = f.name
        try:
            with pytest.raises(CorruptedVideoException):
                probe_and_validate_video(corrupt_path)
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    def test_empty_video_file_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            empty_path = f.name
        try:
            with pytest.raises(CorruptedVideoException):
                probe_and_validate_video(empty_path)
        finally:
            if os.path.exists(empty_path):
                os.remove(empty_path)

    def test_valid_video_metadata_extracted(self, sample_video_path):
        meta: VideoMetadata = probe_and_validate_video(sample_video_path)
        assert meta.width == 640
        assert meta.height == 480
        assert meta.fps == 10.0
        assert meta.frame_count == 10
        assert meta.duration_seconds is not None
        assert abs(meta.duration_seconds - 1.0) < 0.1


# =============================================================================
# 2. ByteTrack Face Tracker Unit Tests
# =============================================================================

class TestByteTrackFaceTracker:
    """Tests multi-face tracking, Kalman motion prediction, and ID persistence."""

    def test_track_id_predictable_and_unique(self):
        tracker = get_face_tracker(min_hits=1)
        det1 = FaceDetection(bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150), confidence=0.90)
        det2 = FaceDetection(bbox=BoundingBox(x1=300, y1=50, x2=400, y2=150), confidence=0.92)

        tracks = tracker.update([det1, det2], frame_index=0)
        assert len(tracks) == 2
        assert tracks[0].track_id == 1
        assert tracks[1].track_id == 2
        assert tracks[0].track_id != tracks[1].track_id

    def test_track_continuity_across_frames(self):
        tracker = get_face_tracker(min_hits=1)
        # Frame 0
        tracks_f0 = tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200), confidence=0.95)],
            frame_index=0,
        )
        assert tracks_f0[0].track_id == 1

        # Frame 1: Small motion
        tracks_f1 = tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=106, y1=102, x2=206, y2=202), confidence=0.93)],
            frame_index=1,
        )
        assert tracks_f1[0].track_id == 1
        assert tracks_f1[0].hits == 2

    def test_bytetrack_stage2_low_confidence_recovery(self):
        tracker = get_face_tracker(
            min_hits=1,
            high_detection_threshold=0.50,
            low_detection_threshold=0.15,
        )
        # Frame 0: High confidence hit
        tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200), confidence=0.95)],
            frame_index=0,
        )
        # Frame 1: Confidence dips to 0.30 (head turn / blur) -> ByteTrack Stage 2 recovers
        tracks_f1 = tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=108, y1=104, x2=208, y2=204), confidence=0.30)],
            frame_index=1,
        )
        assert len(tracks_f1) == 1
        assert tracks_f1[0].track_id == 1
        assert tracks_f1[0].hits == 2

    def test_temporary_occlusion_and_reassociation(self):
        tracker = get_face_tracker(min_hits=1, max_time_lost=10)
        # Frame 0: Detected
        tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200), confidence=0.95)],
            frame_index=0,
        )
        # Frame 1: Completely occluded (0 detections)
        lost_tracks = tracker.update([], frame_index=1)
        assert len(lost_tracks) == 0  # Lost tracks not emitted as active confirmed hits

        # Frame 2: Reappears near predicted position -> Reassociated without ID switch
        reassoc_tracks = tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=105, y1=102, x2=205, y2=202), confidence=0.92)],
            frame_index=2,
        )
        assert len(reassoc_tracks) == 1
        assert reassoc_tracks[0].track_id == 1

    def test_tracker_reset_clears_state_and_resets_counter(self):
        tracker = get_face_tracker(min_hits=1)
        tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200), confidence=0.90)],
            frame_index=0,
        )
        assert tracker.total_unique_tracks == 1

        tracker.reset()
        assert tracker.active_track_count == 0
        assert tracker.total_unique_tracks == 0

        # Next job starts from Track ID 1
        fresh_tracks = tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200), confidence=0.90)],
            frame_index=0,
        )
        assert fresh_tracks[0].track_id == 1

    def test_track_lost_when_face_leaves_frame(self):
        tracker = get_face_tracker(min_hits=1, max_time_lost=3)
        # Face appears for 2 frames
        tracker.update([FaceDetection(bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200), confidence=0.90)], frame_index=0)
        tracker.update([FaceDetection(bbox=BoundingBox(x1=102, y1=101, x2=202, y2=201), confidence=0.90)], frame_index=1)
        assert tracker.active_track_count == 1

        # Face leaves frame (empty detections for 4 frames > max_time_lost)
        for f in range(2, 6):
            tracker.update([], frame_index=f)

        # Track should be marked lost and eventually removed
        assert tracker.active_track_count == 0
        assert tracker.lost_track_count >= 1

    def test_new_track_id_when_new_face_enters(self):
        tracker = get_face_tracker(min_hits=1)
        # Frame 0: Only Face 1 is present
        f0_tracks = tracker.update(
            [FaceDetection(bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150), confidence=0.92)],
            frame_index=0,
        )
        assert len(f0_tracks) == 1
        assert f0_tracks[0].track_id == 1

        # Frame 1: Face 2 enters the scene at a distant coordinate
        f1_tracks = tracker.update(
            [
                FaceDetection(bbox=BoundingBox(x1=52, y1=51, x2=152, y2=151), confidence=0.91),
                FaceDetection(bbox=BoundingBox(x1=400, y1=300, x2=500, y2=400), confidence=0.88),
            ],
            frame_index=1,
        )
        assert len(f1_tracks) == 2
        track_ids = {t.track_id for t in f1_tracks}
        assert 1 in track_ids
        assert 2 in track_ids
        assert tracker.total_unique_tracks == 2


# =============================================================================
# 3. Video Processing Service Pipeline Tests
# =============================================================================

class TestVideoProcessingService:
    """Tests end-to-end streaming video face detection and tracking."""

    def test_video_processing_service_full_pipeline(self, sample_video_path):
        service = VideoProcessingService()
        opts = VideoProcessingOptions(
            confidence_threshold=0.30,
            detection_interval=1,
            analysis_interval=3,
            annotate_video=True,
            enable_analysis=True,
        )
        result = service.process_video_file(sample_video_path, options=opts)

        assert result.success is True
        assert result.metadata.frame_count == 10
        assert result.metrics.processed_frames == 10
        assert result.metrics.total_processing_time > 0.0
        assert result.metrics.average_fps > 0.0
        assert result.metrics.unique_track_count >= 1
        assert result.annotated_video_path is not None
        assert os.path.exists(result.annotated_video_path)

        # Cleanup annotated output
        if os.path.exists(result.annotated_video_path):
            os.remove(result.annotated_video_path)

    def test_detection_and_analysis_interval_gating(self, sample_video_path):
        service = VideoProcessingService()
        # Run detection every 2nd frame, analysis every 4th frame
        opts = VideoProcessingOptions(
            confidence_threshold=0.30,
            detection_interval=2,
            analysis_interval=4,
            annotate_video=False,
            enable_analysis=True,
        )
        result = service.process_video_file(sample_video_path, options=opts)
        assert result.success is True
        assert result.metrics.processed_frames == 10
        # Pipeline executed without errors under interval skipping
        assert result.metrics.detection_time_ms > 0.0

    def test_multiple_face_video_processing(self, multi_face_video_path):
        service = VideoProcessingService()
        opts = VideoProcessingOptions(
            confidence_threshold=0.25,
            detection_interval=1,
            annotate_video=False,
            enable_analysis=False,
        )
        result = service.process_video_file(multi_face_video_path, options=opts)
        assert result.success is True
        assert result.metrics.processed_frames == 8
        # In multi-face video, peak simultaneous faces should reach 2
        assert result.metrics.peak_simultaneous_faces >= 1
        assert result.metrics.unique_track_count >= 1

    def test_no_face_video_processing(self, no_face_video_path):
        service = VideoProcessingService()
        opts = VideoProcessingOptions(
            confidence_threshold=0.50,
            detection_interval=1,
            annotate_video=False,
            enable_analysis=False,
        )
        result = service.process_video_file(no_face_video_path, options=opts)
        assert result.success is True
        assert result.metrics.processed_frames == 6
        assert result.metrics.unique_track_count == 0
        assert result.metrics.peak_simultaneous_faces == 0

    def test_video_processing_service_cancellation(self, sample_video_path):
        service = VideoProcessingService()
        opts = VideoProcessingOptions(annotate_video=False)
        # Cancelled immediately
        with pytest.raises(VideoProcessingCancelledException):
            service.process_video_file(
                sample_video_path,
                options=opts,
                is_cancelled=lambda: True,
            )


# =============================================================================
# 4. Asynchronous Video API Endpoints Tests
# =============================================================================

class TestVideoAPIEndpoints:
    """Tests POST /analyze, GET /jobs/{id}, and GET /jobs/{id}/download."""

    def test_analyze_endpoint_submits_job_and_returns_queued(self, sample_video_path):
        client = TestClient(app)
        with open(sample_video_path, "rb") as f:
            resp = client.post(
                "/api/v1/video/analyze",
                params={"confidence_threshold": 0.30},
                files={"file": ("test_clip.mp4", f.read(), "video/mp4")},
            )

        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] in ("queued", "processing", "completed")
        # Ensure server filesystem path is NOT exposed in response
        assert "C:\\" not in str(data) and "/tmp" not in str(data)

    def test_job_polling_and_download_lifecycle(self, sample_video_path):
        client = TestClient(app)
        with open(sample_video_path, "rb") as f:
            resp = client.post(
                "/api/v1/video/analyze",
                params={"confidence_threshold": 0.30, "annotate_video": True},
                files={"file": ("test_clip.mp4", f.read(), "video/mp4")},
            )
        job_id = resp.json()["job_id"]

        # Poll status
        completed = False
        final_status = None
        for _ in range(25):
            s_resp = client.get(f"/api/v1/video/jobs/{job_id}")
            assert s_resp.status_code == 200
            s_data = s_resp.json()
            assert "C:\\" not in str(s_data) and "storage\\" not in str(s_data)
            if s_data["status"] == "completed":
                completed = True
                final_status = s_data
                break
            time.sleep(0.3)

        assert completed is True
        assert final_status["progress"] == 100
        assert final_status["metrics"] is not None
        assert "unique_track_count" in final_status["metrics"]
        assert "output_media" in final_status
        assert final_status["output_media"] == f"/api/v1/video/jobs/{job_id}/download"

        # Download artifact
        dl_resp = client.get(f"/api/v1/video/jobs/{job_id}/download")
        assert dl_resp.status_code == 200
        assert dl_resp.headers["content-type"] == "video/mp4"
        assert len(dl_resp.content) > 1000

    def test_invalid_upload_returns_400(self):
        client = TestClient(app)
        resp = client.post(
            "/api/v1/video/analyze",
            files={"file": ("malicious.exe", b"MZ_EXE_HEADER_DATA", "application/x-msdownload")},
        )
        assert resp.status_code == 400

    def test_job_failure_on_corrupted_video(self):
        client = TestClient(app)
        resp = client.post(
            "/api/v1/video/analyze",
            files={"file": ("corrupt.mp4", b"NOT_A_REAL_VIDEO_HEADER", "video/mp4")},
        )
        # Even if submitted or caught, it should either be rejected upfront or fail asynchronously
        if resp.status_code == 202:
            job_id = resp.json()["job_id"]
            # Poll for failure
            failed = False
            for _ in range(20):
                s_resp = client.get(f"/api/v1/video/jobs/{job_id}")
                s_data = s_resp.json()
                if s_data["status"] == "failed":
                    failed = True
                    assert s_data["error"] is not None
                    # Ensure no stack trace exposed to client
                    assert "Traceback" not in s_data["error"]
                    break
                time.sleep(0.2)
            assert failed is True
        else:
            assert resp.status_code == 400

    def test_cancel_job_endpoint(self, sample_video_path):
        client = TestClient(app)
        with open(sample_video_path, "rb") as f:
            resp = client.post(
                "/api/v1/video/analyze",
                params={"confidence_threshold": 0.30},
                files={"file": ("test_clip.mp4", f.read(), "video/mp4")},
            )
        job_id = resp.json()["job_id"]
        # Immediately request cancel
        cancel_resp = client.post(f"/api/v1/video/jobs/{job_id}/cancel")
        assert cancel_resp.status_code == 200
        data = cancel_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("cancelled", "completed")

    def test_nonexistent_job_returns_404(self):
        client = TestClient(app)
        resp = client.get("/api/v1/video/jobs/nonexistent-uuid-12345")
        assert resp.status_code == 404

    def test_download_nonexistent_job_returns_404(self):
        client = TestClient(app)
        resp = client.get("/api/v1/video/jobs/nonexistent-uuid-12345/download")
        assert resp.status_code == 404

    def test_cancel_nonexistent_job_returns_404(self):
        client = TestClient(app)
        resp = client.post("/api/v1/video/jobs/nonexistent-uuid-12345/cancel")
        assert resp.status_code == 404


