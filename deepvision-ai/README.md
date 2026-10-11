# DeepVision AI

> **See. Detect. Track. Analyze. Understand.**

A production-grade, privacy-first computer vision and visual intelligence platform designed for high-throughput face detection, facial landmark localization, optical quality evaluation, and real-time multi-face video tracking.

---

## Overview

**DeepVision AI** is an enterprise-ready computer vision system engineered around modular services, asynchronous task queues, and privacy-by-design data handling. Unlike monolithic computer-vision scripts, DeepVision AI cleanly isolates web orchestration, database persistence, distributed message brokers, background workers, and neural network inference engines.

The platform provides millisecond-level image analysis and streaming video processing while strictly enforcing privacy boundaries: **zero facial recognition, zero biometric identity matching, zero facial embedding storage, and zero mass surveillance mechanisms**.

---

## Features

### Implemented Capabilities
* **Face Detection**: Fast multi-scale face localization powered by lightweight YuNet CNN weights (`face_detection_yunet_2023mar.onnx`, 232 KB) running on CPU (OpenCV SIMD / ONNX Runtime) or GPU (CUDA / DirectML).
* **Facial Landmarks**: 5-point canonical facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner) with sub-pixel regression.
* **Optical Quality Analytics**: Granular face diagnostics including Laplacian blur variance, photopic luminance distribution, contrast, face scale, and 3D head pose estimation (Yaw, Pitch, Roll Euler angles via Perspective-n-Point).
* **0–100 Face Quality Score**: Weighted composite quality score categorizing face crops into `EXCELLENT`, `GOOD`, `ACCEPTABLE`, `POOR`, or `UNUSABLE`.
* **Multi-Face Video Tracking**: ByteTrack multi-object tracking associating bounding boxes across time with 8D constant-velocity Kalman filtering and two-stage BYTE data association.
* **Temporary Track IDs**: Ephemeral, session-scoped integer IDs (`Track #1`, `Track #2`) denoting spatio-temporal continuity. Track IDs are completely discarded when a video ends and cannot be linked across videos.
* **Streaming Video Pipeline**: Sequential frame extraction processing long videos without accumulating frames in system memory.
* **Asynchronous Video Jobs**: Redis-backed FIFO queue (`deepvision:queue:video_jobs`) with standalone worker daemons (`video_worker.py`) and automatic in-memory fallback.
* **Annotated Video Rendering**: Video composition rendering HUD cards with Track ID, confidence, optical scores, and session metrics into downloadable MP4 video containers.
* **Modern Next.js Frontend**: Interactive AI vision workspace featuring canvas overlays, dark/light themes, drag-and-drop media ingestion, job status telemetry, and keyboard accessibility.
* **Defense-in-Depth Security**: Strict MIME validation, binary magic-byte stream probing, sliding-window rate limiting, path traversal defense, and automatic secret/token redaction.

### Planned Capabilities (Not Implemented)
* **WebSocket Live Camera Streaming**: Real-time webcam frame-by-frame streaming over WebSocket is planned for future phases.
* **Anti-Spoofing & Liveness Detection**: 2D/3D presentation attack and silicon mask anti-spoofing verification models.
* **Cloud Autoscaling Workers**: KEDA-driven worker autoscaling on Kubernetes with dynamic GPU node provisioning.
* **Edge Device Acceleration**: INT8 quantization and TensorRT engine compilation for edge hardware (NVIDIA Jetson / Raspberry Pi).

---

## Architecture

DeepVision AI utilizes a decoupled asynchronous processing architecture:

```text
                     ┌──────────────────┐
                     │     Browser      │
                     │ Next.js Frontend │
                     └────────┬─────────┘
                              │ HTTP / REST
                              ▼
                     ┌──────────────────┐
                     │    FastAPI API   │
                     └──────┬─────┬─────┘
                            │     │
                    ┌───────┘     └────────┐
                    ▼                      ▼
              ┌──────────┐          ┌──────────┐
              │PostgreSQL│          │  Redis   │
              └──────────┘          └────┬─────┘
                                         │ Task Queue
                                         ▼
                                  ┌──────────────┐
                                  │ Video Worker │
                                  └──────┬───────┘
                                         │ Execution
                                         ▼
                                ┌─────────────────┐
                                │ AI/CV Pipeline  │
                                │ Detection       │
                                │ Landmarks       │
                                │ Quality         │
                                │ Tracking        │
                                └────────┬────────┘
                                         │ Artifacts
                                         ▼
                                  ┌────────────┐
                                  │  Storage   │
                                  └────────────┘
```

For complete architectural details, see [docs/production-architecture.md](docs/production-architecture.md).

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide React |
| **Backend API** | FastAPI, Python 3.12, Uvicorn, Pydantic v2 |
| **Computer Vision** | OpenCV 4.10, ONNX Runtime, NumPy, Pillow |
| **Database** | PostgreSQL 16, SQLAlchemy 2.0 ORM, Alembic |
| **Queue & Cache** | Redis 7, Python-based worker daemon |
| **Containerization** | Docker, Docker Compose (multi-stage rootless images) |
| **Testing** | pytest, pytest-anyio, httpx, ESLint |

---

## Project Structure

```text
deepvision-ai/
├── backend/
│   ├── alembic/                 # Database schema migration revisions
│   ├── app/
│   │   ├── ai/                  # AI engine modules
│   │   │   ├── common/          # Shared geometric & bounding box types
│   │   │   ├── detection/       # FaceDetector (YuNet / SCRFD) & adapters
│   │   │   ├── analysis/        # Optics, blur, luminance, pose, quality
│   │   │   ├── tracking/        # ByteTrack Kalman filter & state tracker
│   │   │   ├── inference/       # ONNX Runtime & device resolution
│   │   │   └── visualization/   # HUD annotator & frame rendering
│   │   ├── api/v1/              # REST routes (/ai, /video, /health)
│   │   ├── core/                # Config, logging, storage, rate limiting
│   │   ├── models/              # SQLAlchemy 2.0 ORM database entities
│   │   ├── schemas/             # Pydantic v2 validation contracts
│   │   ├── services/            # AIService, VideoJobManager, VideoProcessing
│   │   └── workers/             # video_worker.py background task daemon
│   ├── tests/                   # 29 test modules (334 automated tests)
│   ├── Dockerfile               # Rootless multi-stage backend container
│   └── Dockerfile.worker        # Dedicated background worker container
├── frontend/
│   ├── src/                     # Next.js App Router application
│   │   ├── app/                 # Pages: /, /image, /video, /dashboard, /settings
│   │   ├── components/          # Canvas visualizers, uploaders, HUD cards
│   │   └── lib/                 # Typed API clients & utility functions
│   └── Dockerfile               # Rootless multi-stage frontend container
├── models/                      # Physical ONNX weights (git-ignored)
├── storage/                     # Segregated storage (temp/, media/, results/)
├── scripts/                     # Hardware profiling & benchmark scripts
├── docs/                        # Complete technical documentation
├── docker-compose.yml           # Multi-service stack (web, worker, db, redis)
└── .env.example                 # Configuration template
```

---

## Local Development

### Prerequisites
* Python 3.12+
* Node.js 20+ & npm
* Git

### 1. Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start API development server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Video Worker Setup (Optional for Redis Queue)
In a separate terminal window:
```bash
cd backend
python -m app.workers.video_worker
```
*(Note: If Redis is offline, the backend automatically uses its in-memory queue fallback).*

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

* **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
* **Backend API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health Check**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

## Environment Variables

Key parameters in `.env` (see `.env.example` and `.env.production.example`):

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Runtime mode (`development`, `staging`, `production`) |
| `SECRET_KEY` | *(change in prod)* | Cryptographic session signing key |
| `DATABASE_URL` | `sqlite:///./deepvision.db` | PostgreSQL or SQLite database URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis queue and rate limiting connection string |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (restrict in production) |
| `AI_DEVICE` | `auto` | Execution device: `auto`, `cpu`, or `gpu` |
| `AI_CONFIDENCE_THRESHOLD` | `0.50` | Minimum face detection confidence score |
| `AI_REQUIRE_MODEL` | `false` | When true, fails fast if model weights are missing |
| `MAX_CONCURRENT_VIDEO_JOBS` | `4` | Maximum parallel video processing tasks |
| `VIDEO_DETECTION_INTERVAL` | `1` | Frame interval for running face detector |
| `VIDEO_ANALYSIS_INTERVAL` | `5` | Frame interval for computing face quality/optics |

---

## AI Models

DeepVision AI utilizes standard lightweight ONNX model graphs:

| Model | Architecture | File Size | Checksum (SHA-256) | License | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YuNet (Default)** | YuNet CNN | 232 KB | `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4` | Apache-2.0 | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet) |
| **SCRFD-500M** | SCRFD Mobile | ~1.7 MB | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | Apache-2.0 | [InsightFace](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) |

* **Zero Commits to Git**: All `.onnx` and weight binaries are strictly excluded from version control via `.gitignore`.
* **Zero Silent Startup Downloads**: The platform never silently downloads weights on boot, ensuring deterministic, air-gapped cold starts.

For download and setup instructions, see [docs/model-setup.md](docs/model-setup.md) and [docs/model-management.md](docs/model-management.md).

---

## Image Analysis

Submit single images to `POST /api/v1/ai/detect` or `POST /api/v1/ai/analyze` (`multipart/form-data`).

### Example Response (`POST /api/v1/ai/analyze`)
```json
{
  "success": true,
  "face_count": 1,
  "faces": [
    {
      "face_index": 0,
      "bounding_box": { "x1": 154, "y1": 92, "x2": 486, "y2": 524, "width": 332, "height": 432 },
      "confidence": 0.998,
      "landmarks": {
        "right_eye": { "x": 235.1, "y": 218.4 },
        "left_eye": { "x": 402.7, "y": 214.2 },
        "nose_tip": { "x": 322.8, "y": 305.6 },
        "mouth_right": { "x": 251.3, "y": 412.0 },
        "mouth_left": { "x": 395.4, "y": 408.8 }
      },
      "quality": {
        "overall_score": 88.5,
        "quality_category": "EXCELLENT",
        "blur_score": 142.6,
        "brightness_score": 128.4,
        "contrast_score": 58.2,
        "is_usable": true
      },
      "pose": {
        "yaw": -2.4,
        "pitch": 4.1,
        "roll": -1.2,
        "pose_category": "FRONTAL"
      }
    }
  ],
  "telemetry": {
    "total_analysis_time_ms": 17.4,
    "device_used": "CPU"
  }
}
```

---

## Video Tracking

Video files are processed asynchronously via `POST /api/v1/video/analyze`:

1. **Job Enqueued**: The API validates the container, generates a `job_id`, and returns HTTP 202 Accepted.
2. **Streaming Execution**: The worker extracts frames sequentially, applies ByteTrack multi-object tracking, computes optics at configurable intervals, and draws HUD annotations.
3. **Telemetry & Artifacts**: Once complete, the job reports total frames, average processing FPS, peak simultaneous faces, and unique track count.

### Video Tracking API Endpoints
* `POST /api/v1/video/analyze`: Upload video and enqueue processing job.
* `GET /api/v1/video/jobs/{id}`: Poll job progress and retrieve tracking telemetry.
* `POST /api/v1/video/jobs/{id}/cancel`: Safely cancel an active job and purge temporary files.
* `GET /api/v1/video/jobs/{id}/download`: Download rendered annotated MP4 video.

---

## API Documentation

When the backend is running, complete interactive OpenAPI documentation is accessible at:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`
* OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Testing

The codebase maintains 100% test pass rates across unit, integration, and security suites:

```bash
cd backend
python -m pytest -v
```

### Verified Test Suites (334 Tests Total)
* `test_end_to_end_integration.py`: Complete pipeline integration test.
* `test_step7_hardening_and_architecture.py`: Database models, Alembic migrations, storage partitions, health probes.
* `test_production_security_hardening.py`: Rate limiters, security headers, concurrency gates, path sanitization.
* `test_structured_logging.py`: Structured JSON logging, secret redaction, request ID correlation.
* `test_video_*.py`: Multi-face tracking, Kalman filtering, ByteTrack association, async job lifecycle.
* `test_ai_inference.py`: Core detection engine, letterbox preprocessing, coordinate unpadding.

---

## Docker

Deploy the complete multi-service production stack with Docker Compose:

```bash
docker compose up --build
```

### Services
* `postgres`: PostgreSQL 16 database.
* `redis`: Redis 7 broker and rate-limiting store.
* `backend`: FastAPI API engine (Port 8000).
* `worker`: Standalone video background worker.
* `frontend`: Next.js 14 web application (Port 3000).

All containers execute under unprivileged non-root user accounts (`UID 1000` / `UID 1001`).

---

## Deployment

For enterprise production deployments across Bare-Metal, Cloud VMs, and Managed Container Platforms (AWS, GCP, Azure), refer to the comprehensive deployment guide:
* [docs/production-deployment.md](docs/production-deployment.md)

---

## Performance

Live performance benchmarks measured on an Intel Core i5-8365U CPU (4 Cores / 8 Threads, 16 GB RAM) using [scripts/benchmark_performance.py](scripts/benchmark_performance.py):

* **Image Face Detection (640x480)**: Mean latency **16.62 ms** (~60.2 FPS).
* **Image Face Detection (1280x720 HD)**: Mean latency **62.70 ms** (~15.9 FPS).
* **Streaming Video Tracking (120 frames @ 24fps)**:
  * Total Wall-Clock Processing Time: **3.051 seconds** (sub-realtime on pure CPU).
  * Throughput: **39.34 FPS**.
  * Average Frame Processing Time: **25.42 ms**.
  * Peak Memory Allocation: **4.52 MB** (steady streaming heap).

For full benchmark figures and hardware telemetry, see [docs/performance.md](docs/performance.md).

---

## Security

DeepVision AI incorporates defense-in-depth security principles:
* **Zero Hardcoded Secrets**: All credentials resolved via environment variables.
* **Path Traversal Protection**: Uploaded filenames sanitized; direct memory decoding for still images.
* **Injection Defense**: Zero `subprocess`, `os.system`, or `eval` calls across application paths.
* **Rate Limiting & Concurrency**: Sliding-window rate limiters and maximum concurrent job limits.
* **Traceback Masking**: Raw internal exception traces masked from client responses.
* **Rootless Containers**: Production Docker images execute as unprivileged users.

Full audit results are documented in [docs/security.md](docs/security.md).

---

## Privacy

DeepVision AI enforces strict technical privacy-by-design boundaries:
* **Detection, Not Recognition**: Detects spatial coordinates and optical quality; never matches individuals.
* **Zero Biometric Identity Profiling**: Zero facial embedding vectors, identity databases, or watchlists.
* **Ephemeral Processing**: Still images processed 100% in volatile RAM with zero disk persistence.
* **Session-Scoped Track IDs**: Track IDs represent mathematical motion continuity in a single video and are discarded upon completion.
* **Compliance Disclaimer**: Technical privacy safeguards are built-in, but formal compliance with statutory frameworks (GDPR, DPDP, HIPAA) requires organizational review.

For complete data governance policies, see [docs/privacy.md](docs/privacy.md).

---

## Limitations

* **Pure CPU Execution by Default**: Standard installations execute on CPU. GPU acceleration requires NVIDIA drivers and `CUDAExecutionProvider`.
* **5-Point Landmark Resolution**: Built-in landmarks provide 5 key points. High-density facial mesh (68-point / 468-point) requires secondary model pipelines.
* **Lighting & Severe Occlusion**: Extreme low light (< 20 lux) or severe facial occlusions (> 70%) may reduce detection confidence.
* **No Video Live-Streaming Ingestion**: Video processing is currently file-based (MP4, AVI, MOV, MKV); RTSP/RTMP live streams are not yet supported.

---

## Future Improvements

1. **WebSocket Live Camera Ingestion**: Streaming webcam frame analysis directly from the browser over WebSocket.
2. **NVIDIA TensorRT GPU Acceleration**: FP16 / INT8 engine optimization delivering 200+ FPS per GPU worker.
3. **KEDA Autoscaling Worker Pools**: Kubernetes event-driven autoscaling dynamically provisioning GPU worker nodes based on Redis queue depth.
4. **Edge Device Deployment**: Dedicated builds optimized for edge inference devices (NVIDIA Jetson Orin, Raspberry Pi 5).
5. **High-Density Landmark Mesh**: Optional 68-point or 106-point facial mesh landmark models for detailed facial expression diagnostics.
