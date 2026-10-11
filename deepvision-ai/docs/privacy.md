# DeepVision AI — Privacy Architecture & Data Governance

This document describes the privacy architecture, data handling practices, storage lifecycles, and identity boundaries implemented in DeepVision AI (Step 7.24).

---

## 1. Principles of Privacy by Design

DeepVision AI is architected from the ground up to analyze visual optics and localize faces **without compromising personal privacy or establishing biometric identity profiles**:

1. **Detection, Not Recognition**: The platform detects spatial coordinates of faces and measures optical attributes (sharpness, illumination, pose, visibility). It does **not** recognize, identify, or match individuals.
2. **Zero Biometric Identity Profiling**: The system does not compute, store, or index facial embedding vectors, biometric facial templates, or biometric identifiers.
3. **Transient Processing by Default**: Media inputs are treated as transient payloads; raw images are processed entirely in memory with zero disk persistence.
4. **Ephemerality of Tracking**: Multi-face tracking is confined strictly to the context of a single video session. Track IDs cannot be linked across separate videos or user sessions.

---

## 2. What Media is Processed

DeepVision AI processes visual media submitted directly by authenticated users or authorized clients:

* **Digital Still Images**: Standard photographic formats (`image/jpeg`, `image/png`, `image/webp`).
* **Digital Video Containers**: Standard video streams (`video/mp4`, `video/avi`, `video/mov`, `video/mkv`).

The platform extracts visual frames solely to perform:
* Bounding box localization (face position within frame).
* 5-point facial landmark geometry (eyes, nose tip, mouth corners) used strictly for alignment and head pose estimation.
* Optical quality diagnostics (Laplacian blur variance, luminance histogram, contrast, face size).

---

## 3. Storage Architecture & Deletion Lifecycles

### 3.1 Still Images: Zero Disk Persistence
* **Processing Location**: Volatile RAM only.
* **Storage Path**: None. Still images uploaded to `POST /api/v1/ai/detect` and `POST /api/v1/ai/analyze` are decoded directly from incoming HTTP byte streams into in-memory NumPy arrays.
* **Deletion**: Once inference and analysis complete, the in-memory array is garbage-collected. No raw image file is written to physical disk storage or object buckets.

### 3.2 Videos: Scoped Ephemeral Storage
* **Ingestion Storage**: Uploaded video files are stored in `storage/temp/` with cryptographically random UUIDv4 filenames.
* **Streaming Processing**: Frames are decoded sequentially via OpenCV `VideoCapture`. At no point is the entire video loaded into memory.
* **Scratch Cleanup**: Intermediate frame caches and temporary video files are unlinked and permanently deleted upon pipeline completion or error via deterministic `finally` blocks.
* **Annotated Video Artifacts**: When output annotation is enabled, the rendered video is saved to `storage/results/` so the client can download it via `GET /api/v1/video/jobs/{id}/download`.

---

## 4. Analysis Results & Database Persistence

The relational database (`PostgreSQL` / `SQLite`) records structured metadata, job lifecycle states, and analytical scores:

* **What IS Persisted**:
  * Job metadata: `job_id`, `created_at`, `status`, `processing_time_ms`.
  * Aggregate counts: `total_frames`, `processed_frames`, `face_count`, `unique_track_count`.
  * Optical metrics: blur scores, brightness values, head pose angles (pitch, yaw, roll).
  * Trajectory summaries: duration of tracks, peak concurrent faces.
* **What is NEVER Persisted**:
  * Raw facial images or cropped face chips.
  * 128-dimensional or 512-dimensional facial recognition embedding vectors.
  * Personal identifying information (PII) mapped to facial structures.
  * Facial biometric comparison databases or watchlist tables.

---

## 5. Track ID Semantics

In video processing, faces are assigned sequential tracking labels (e.g., `Track #1`, `Track #2`):

* **Spatial-Temporal Continuity Only**: A Track ID indicates that a localized bounding box in frame $N$ likely corresponds to the continuous trajectory of the bounding box in frame $N+1$ using Kalman filter motion modeling and Intersection over Union (IoU) association.
* **No Persistent Identity**: Track IDs have zero relationship to a human individual's real identity. They represent trajectory slots, not persons.
* **Session-Isolated**: When a video processing job completes, its tracking state is destroyed. Track IDs are completely non-transferable across different video files.

---

## 6. User Control Over Media & Cancellation

Users maintain full control over media submissions:

* **Active Cancellation**: An active video processing job can be cancelled at any point via `POST /api/v1/video/jobs/{id}/cancel`. Cancellation immediately terminates worker execution, discards in-progress output streams, and unlinks temporary files.
* **Artifact Retrieval & Purging**: Clients can configure retention windows for rendered artifacts in `storage/results/` or trigger administrative cleanup routines.

---

## 7. Regulatory & Compliance Notice

DeepVision AI implements rigorous technical privacy safeguards and architectural data minimization. However:

> [!IMPORTANT]
> This platform makes **no formal claim of compliance** with specific legal or regulatory frameworks (including GDPR, EU AI Act, California CCPA, Indian DPDP, or HIPAA). Compliance with statutory data protection regulations depends on the deploying organization's operational governance, consent collection practices, deployment environment, data retention schedules, and legal jurisdiction. Organizations deploying DeepVision AI in production must conduct their own legal and compliance reviews.
