# DeepVision AI — Production Architecture Specification

This document details the production system architecture, process topology, data pipelines, and horizontal scaling strategies for DeepVision AI (Step 7.27).

---

## 1. System Topology & Data Flow

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
                                         │ Queue / Jobs
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

---

## 2. Core Architectural Subsystems

### 2.1 Presentation Layer (Next.js 14)
* **Framework**: Next.js 14 App Router with React Server Components, TypeScript, and Tailwind CSS.
* **Responsibilities**:
  * Interactive Computer Vision Dashboard with Dark/Light theme persistence.
  * Direct canvas rendering of bounding boxes, 5-point facial landmark constellations, and head pose orientation vectors.
  * Real-time polling and progress indicator for asynchronous video tracking jobs.
  * Client-side validation: MIME type checks and payload size pre-flight validation.

### 2.2 API Gateway & Ingestion Layer (FastAPI)
* **Framework**: Python 3.12 + FastAPI + Uvicorn + Starlette.
* **Responsibilities**:
  * Enforcing security headers (`HSTS`, `X-Frame-Options`, `nosniff`, `strict-origin`).
  * In-memory sliding-window rate limiting on compute-heavy routes (`/ai/detect`, `/video/analyze`).
  * Correlation tracking via `X-Request-ID` middleware.
  * Synchronous processing of single-image detection and multi-attribute analysis requests in volatile memory (zero disk persistence).
  * Video job admission control: validating container headers, generating UUIDv4 job tokens, recording pending job entries in PostgreSQL, and enqueuing job IDs into Redis.

### 2.3 Persistence Layer (PostgreSQL)
* **Engine**: PostgreSQL 16 managed via SQLAlchemy 2.0 ORM and Alembic migrations.
* **Data Schema**:
  * `users`: Authentication and authorization credentials.
  * `analysis_jobs`: Lifecycle tracking (`queued`, `processing`, `completed`, `failed`, `cancelled`), progress counters, and execution timestamps.
  * `analysis_results`: Aggregate telemetry, face counts, optical metrics, and trajectory summaries.
  * `media_assets`: Metadata pointers for persisted artifacts in object storage.
  * `system_logs`: Audit events, request durations, and operational logs.
* **Privacy Boundary**: Zero biometric identity templates, zero facial recognition embeddings, and zero raw image pixel blobs are stored in SQL tables.

### 2.4 Distributed Message Broker & Cache (Redis)
* **Engine**: Redis 7.
* **Responsibilities**:
  * Task Queue: `deepvision:queue:video_jobs` list for FIFO job scheduling.
  * State Store: `deepvision:job:<id>` hashes storing atomic progress counters, frame status, and worker heartbeats.
  * Concurrency Guard: Atomic sets enforcing `MAX_CONCURRENT_VIDEO_JOBS` cluster limits.
  * Fallback: Decoupled architecture automatically engages `InMemoryVideoJobManager` if Redis is offline or disabled.

### 2.5 Background Asynchronous Worker (`video_worker.py`)
* **Process**: Standalone Python daemon decoupled from API web workers.
* **Lifecycle**:
  * Pops jobs atomically via `BLPOP` on `deepvision:queue:video_jobs`.
  * Checks for cancellation flags before and during streaming frame extraction.
  * Updates frame counters and progress percentage in Redis/PostgreSQL.
  * Executes cleanup routines in `finally` blocks to guarantee zero orphaned scratch files.
  * Emits heartbeat signals to Redis for worker liveness tracking.

### 2.6 Computer Vision & Tracking Engine
* **Detection**: OpenCV DNN SIMD backend / ONNX Runtime running lightweight YuNet CNN weights (`face_detection_yunet_2023mar.onnx`, 232 KB).
* **Landmarks & Alignment**: 5 canonical landmarks (eyes, nose, mouth corners) with eye-tilt angle, inter-pupillary distance, and facial symmetry.
* **Optics & Diagnostics**: Laplacian sharpness variance, luminance histogram distribution, and perspective-n-point (PnP) 3D head pose estimation.
* **Tracking**: ByteTrack with 8D constant-velocity Kalman filter associating bounding boxes across time with ephemeral, non-persistent Track IDs.

### 2.7 Storage Subsystem (`LocalStorage` & `ObjectStorage`)
* **Partitioning**:
  * `storage/temp/`: Ephemeral scratch space for chunked video processing.
  * `storage/media/`: Uploaded source media assets.
  * `storage/results/`: Rendered MP4 annotated videos available for client download.
* **Abstraction**: Pluggable storage interface supporting local directories with automated cloud emulation fallback for AWS S3 and Google Cloud Storage.

---

## 3. Scaling to Cloud GPU Worker Pools

The decoupled architecture allows scaling compute-heavy video processing without altering web APIs or database layers:

```text
               ┌───────────────────────┐
               │    FastAPI Web Pod    │ (Autoscaled on HTTP traffic)
               └───────────┬───────────┘
                           │ Enqueue Job
                           ▼
               ┌───────────────────────┐
               │   Redis Queue Cluster │
               └───────────┬───────────┘
                           │ Dequeue Job
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      ┌────────────┐┌────────────┐┌────────────┐
      │ GPU Worker ││ GPU Worker ││ GPU Worker │ (Autoscaled on Queue Depth)
      │ (NVIDIA T4)││(NVIDIA A10)││(NVIDIA L4) │
      └────────────┘└────────────┘└────────────┘
```

### 3.1 Scaling Strategy
1. **Independent Worker Autoscaling**:
   - Web API pods scale horizontally based on HTTP CPU utilization and request rates.
   - Worker pods scale based on Redis queue depth (`deepvision:queue:video_jobs` length) using Kubernetes KEDA or AWS Target Tracking.
2. **GPU Acceleration Activation**:
   - In worker containers, set `AI_ENABLE_GPU=true` and `AI_DEVICE=cuda`.
   - The underlying `InferenceRuntime` automatically engages ONNX Runtime's `CUDAExecutionProvider` or `TensorrtExecutionProvider`.
   - On NVIDIA T4 / A10 GPUs, inference latency drops from ~16 ms to ~2–4 ms per frame, enabling 150+ FPS video processing throughput per worker.
3. **Decoupled Shared Storage**:
   - Local directory storage transitions to high-throughput cloud object storage (AWS S3, Google Cloud Storage, Azure Blob Storage) using pre-signed upload and download URLs.
   - Workers download source clips, process frames in memory, and stream rendered MP4 artifacts directly to object buckets.
4. **Spot / Preemptible Instance Resiliency**:
   - Workers report heartbeats and frame milestones every $N$ frames to Redis.
   - If a spot GPU instance is terminated, the coordinator re-queues the job or resumes processing from the last acknowledged checkpoint.
