"""
DeepVision AI — Step 5 Video Performance Benchmark & Verification Script
Accurately measures pipeline latency, throughput, memory consumption,
and track ID stability under real-world conditions.
"""

import os
import sys
import time
import ctypes
from ctypes import wintypes
import cv2
import numpy as np

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings
from app.schemas.video import VideoProcessingOptions
from app.services.video_processing_service import VideoProcessingService
from fastapi.testclient import TestClient
from app.main import app


# -----------------------------------------------------------------------------
# 1. Native Windows Memory Tracker
# -----------------------------------------------------------------------------
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_peak_memory_mb() -> float:
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
    return 0.0


# -----------------------------------------------------------------------------
# 2. Benchmark Video Generation (Real Human Faces in HD 1280x720, 30 FPS)
# -----------------------------------------------------------------------------
def generate_benchmark_video(video_path: str, width: int = 1280, height: int = 720, fps: float = 30.0, num_frames: int = 90):
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    face1_img = cv2.imread(os.path.join(fixtures_dir, "sample_face.jpg"))
    face2_img = cv2.imread(os.path.join(fixtures_dir, "large_face.jpg"))

    if face1_img is None:
        face1_img = np.full((300, 300, 3), 180, dtype=np.uint8)
    if face2_img is None:
        face2_img = np.full((300, 300, 3), 160, dtype=np.uint8)

    face1_crop = cv2.resize(face1_img, (320, 320))
    face2_crop = cv2.resize(face2_img, (300, 300))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create gradient realistic background
        frame = np.full((height, width, 3), 110, dtype=np.uint8)

        # Face 1: Enters from left and moves smoothly towards center
        f1_x = int(80 + (i / num_frames) * 350)
        f1_y = int(120 + np.sin(i * 0.1) * 20)
        frame[f1_y : f1_y + 320, f1_x : f1_x + 320] = face1_crop

        # Face 2: Present on the right, slowly shifts left
        f2_x = int(850 - (i / num_frames) * 200)
        f2_y = int(150 + np.cos(i * 0.08) * 15)
        frame[f2_y : f2_y + 300, f2_x : f2_x + 300] = face2_crop

        out.write(frame)

    out.release()
    print(f"Generated benchmark video: {video_path} ({width}x{height}, {fps} FPS, {num_frames} frames, {num_frames/fps:.2f}s)")


# -----------------------------------------------------------------------------
# 3. Main Benchmark Execution
# -----------------------------------------------------------------------------
def run_benchmark():
    benchmark_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    video_input = os.path.join(benchmark_dir, "benchmark_hd_input.mp4")
    generate_benchmark_video(video_input, width=1280, height=720, fps=30.0, num_frames=90)

    service = VideoProcessingService()
    options = VideoProcessingOptions(
        confidence_threshold=0.30,
        detection_interval=1,
        analysis_interval=5,
        annotate_video=True,
        enable_analysis=True,
    )

    mem_before = get_peak_memory_mb()
    t_start = time.perf_counter()

    result = service.process_video_file(
        video_input_path=video_input,
        options=options,
    )

    t_end = time.perf_counter()
    mem_after = get_peak_memory_mb()

    total_proc_seconds = t_end - t_start
    meta = result.metadata
    metrics = result.metrics

    print("\n" + "=" * 60)
    print("DEEPVISION AI — STEP 5 VIDEO PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Video Resolution:               {meta.width}x{meta.height}")
    print(f"Video Native FPS:               {meta.fps:.2f}")
    print(f"Video Frame Count:              {meta.frame_count}")
    print(f"Video Duration (seconds):       {meta.duration_seconds:.2f} s")
    print(f"Total Processing Time (seconds):{total_proc_seconds:.3f} s")
    print(f"Average Processing FPS:         {meta.frame_count / total_proc_seconds:.2f} fps")
    print(f"Average Frame Time (ms):        {(total_proc_seconds / meta.frame_count) * 1000.0:.2f} ms")
    print(f"Peak Memory Working Set:        {mem_after:.2f} MB")
    print("-" * 60)
    print(f"Detection Time Total:           {metrics.detection_time_ms / 1000.0:.3f} s ({metrics.detection_time_ms / meta.frame_count:.2f} ms/frame)")
    print(f"Tracking Time Total:            {metrics.tracking_time_ms / 1000.0:.3f} s ({metrics.tracking_time_ms / meta.frame_count:.2f} ms/frame)")
    print(f"Analysis Time Total:            {metrics.analysis_time_ms / 1000.0:.3f} s")
    print(f"Annotation Time Total:          {metrics.annotation_time_ms / 1000.0:.3f} s")
    print("-" * 60)
    print(f"Peak Simultaneous Faces:        {metrics.peak_simultaneous_faces}")
    print(f"Unique Track Count:             {metrics.unique_track_count}")
    print(f"Track Lost Count:               {metrics.track_lost_count}")
    print(f"Total Detections Across Frames: {metrics.total_faces_detected}")

    # Inspect track stability
    track_ids_observed = set()
    frames_per_track = {}
    for fr in result.frame_results:
        for t in fr.tracks:
            track_ids_observed.add(t.track_id)
            frames_per_track[t.track_id] = frames_per_track.get(t.track_id, 0) + 1

    print(f"Observed Track IDs:             {sorted(list(track_ids_observed))}")
    for tid, cnt in sorted(frames_per_track.items()):
        print(f"  Track ID {tid}: present in {cnt}/{meta.frame_count} frames ({cnt/meta.frame_count*100:.1f}%)")
    print("=" * 60)

    # -----------------------------------------------------------------------------
    # 4. Asynchronous API End-to-End Verification
    # -----------------------------------------------------------------------------
    print("\nVerifying Asynchronous API Endpoints with Real Video...")
    client = TestClient(app)
    with open(video_input, "rb") as f:
        resp = client.post(
            "/api/v1/video/analyze",
            params={"confidence_threshold": 0.30, "annotate_video": True},
            files={"file": ("benchmark_hd_input.mp4", f.read(), "video/mp4")},
        )
    assert resp.status_code == 202, f"POST /analyze failed: {resp.status_code} {resp.text}"
    job_id = resp.json()["job_id"]
    print(f"Job created successfully: {job_id} (Status: {resp.json()['status']})")

    # Poll status
    done = False
    for attempt in range(40):
        time.sleep(0.5)
        st_resp = client.get(f"/api/v1/video/jobs/{job_id}")
        assert st_resp.status_code == 200
        st_data = st_resp.json()
        print(f"  Poll {attempt+1}: Status={st_data['status']}, Progress={st_data['progress']}%, Processed={st_data['processed_frames']}/{st_data['total_frames']}")
        if st_data["status"] == "completed":
            done = True
            break
        elif st_data["status"] == "failed":
            raise RuntimeError(f"Job failed: {st_data.get('error')}")

    assert done, "Job did not complete within timeout"
    print(f"Job completed with metrics: {st_data.get('metrics')}")

    # Download verification
    dl_resp = client.get(f"/api/v1/video/jobs/{job_id}/download")
    assert dl_resp.status_code == 200, f"Download failed: {dl_resp.status_code}"
    assert dl_resp.headers["content-type"] == "video/mp4"
    assert len(dl_resp.content) > 10000
    print(f"Annotated video downloaded successfully ({len(dl_resp.content)} bytes).")

    # Cleanup input
    if os.path.exists(video_input):
        os.remove(video_input)
    if result.annotated_video_path and os.path.exists(result.annotated_video_path):
        os.remove(result.annotated_video_path)

    print("\nBenchmark and End-to-End API verification completed successfully!")


if __name__ == "__main__":
    run_benchmark()
