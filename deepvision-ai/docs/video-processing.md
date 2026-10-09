# Video Face Detection & Multi-Face Tracking Pipeline

**DeepVision AI — Video Processing Architecture & Technical Specification**  
**Author**: DeepVision AI Computer Vision Engineering Team  
**Status**: ACTIVE  
**Milestone**: Step 5 (Video Face Detection & Multi-Face Tracking)  

---

## 1. Overview & Architecture

DeepVision AI provides streaming, memory-efficient video face detection and multi-face tracking. The subsystem extracts video frames sequentially, performs joint face detection via OpenCV YuNet, tracks facial trajectories across time using ByteTrack with a 2D Kalman Filter, and overlays development telemetry and track cards onto an annotated output video.

### End-to-End Pipeline

```
Video File Upload (MP4, AVI, MOV, MKV, WebM)
      ↓
Security & Container Validation (video_security.py)
      ↓
Stream Metadata Extraction (width, height, fps, duration)
      ↓
Streaming Frame Extraction (cap.read() — zero full-video RAM buffering)
      ↓
Face Detection (YuNet FaceDetector — interval-gated)
      ↓
Multi-Face Tracking (ByteTrackFaceTracker — 2D Kalman motion model)
      ↓
Face Analysis (FaceAnalysisService — quality/pose cached per track)
      ↓
Frame Annotation (FaceVisualizer — HUD, bounding boxes, Track IDs)
      ↓
Output Video Encoding (cv2.VideoWriter)
      ↓
Processing & Tracking Metrics Telemetry
```

---

## 2. Configurable Performance Intervals (Step 5.11)

To optimize CPU throughput while maintaining trajectory continuity, expensive pipeline stages can be executed at decoupled intervals:

| Setting | Default | Description & Operational Effect |
| :--- | :---: | :--- |
| `VIDEO_DETECTION_INTERVAL` | `1` | Controls how frequently neural face detection executes. When set to `1`, detection runs on every frame. When set to $N > 1$, detector runs every $N$th frame; intermediate frames bypass the neural forward pass and coast existing tracks using Kalman velocity predictions, substantially boosting throughput. |
| `VIDEO_ANALYSIS_INTERVAL` | `5` | Controls how frequently detailed optical diagnostics (Laplacian blur, ITU-R luminance, contrast, size category, 3D pose, quality score) are calculated for confirmed tracks. Diagnostics are cached and reused on intermediate frames, eliminating redundant crop processing. |
| `VIDEO_MAX_FPS` | `None` | Optional target frame rate cap for frame subsampling. When specified, frames exceeding the target rate are skipped before inference. |

---

## 3. Track Lifecycle State Machine (Step 5.14)

```
        ┌────────────────────────────────┐
        │     High-Score Detection       │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │      NEW TRACK (Tentative)     │
        └────────────────┬───────────────┘
                         │ hits >= min_hits (default: 2)
                         ▼
        ┌────────────────────────────────┐
 ┌─────►│     ACTIVE TRACK (Confirmed)   │◄────┐
 │      └────────────────┬───────────────┘     │
 │                       │ missed in frame     │
 │                       ▼                     │
 │      ┌────────────────────────────────┐     │
 │      │       TEMPORARILY LOST         │     │
 │      │  (Coasting via Kalman filter)  │     │
 │      └───────┬────────────────┬───────┘     │
 │              │                │             │
 │ ByteTrack    │                │ missed > max_time_lost
 │ Reassociation│                ▼ (default: 30 frames)
 └──────────────┘       ┌────────────────┐
                        │   TERMINATED   │
                        │   (Deleted)    │
                        └────────────────┘
```

### State Definitions

1. **New Track (`TENTATIVE`)**:
   - Initialized when an unmatched high-confidence detection appears.
   - Must be matched across consecutive frames ($N_{\text{hits}} \ge \text{min\_hits}$) before promotion to avoid false-positive track creation from transient artifacts.
2. **Active Track (`CONFIRMED`)**:
   - Promoted track assigned a sequential temporary `track_id` (`Track 1`, `Track 2`, ...).
   - Kalman filter continuously updates bounding box position, aspect ratio, and velocity.
3. **Temporarily Lost (`LOST`)**:
   - Track missed in the current frame (due to sudden head rotation, extreme lighting change, motion blur, or partial hand occlusion).
   - Position is coasted using constant-velocity Kalman prediction.
   - ByteTrack Stage 2 attempts association with secondary low-confidence detections ($0.15 \le \text{conf} < 0.50$).
4. **Reassociated**:
   - Track is successfully matched again with a detection before expiration, recovering continuity without an ID switch.
5. **Terminated (`DELETED`)**:
   - Track exceeds `max_time_lost` consecutive missed frames (~1 second at 30 fps) and is permanently retired.

---

## 4. Track ID Invariants & Identity Rules (Step 5.3, 5.15)

In compliance with DeepVision AI security controls:

* **Predictable Local Counter**: Track IDs are strictly sequential integers starting from `1` (`Track 1`, `Track 2`, `Track 3`, ...).
* **Job-Scoped Ephemerality**: Track IDs are unique **only within a single video processing job**.
* **Zero Cross-Job Persistence**: Calling `tracker.reset()` or launching a new video job resets the counter back to `1`.
* **Zero Identity Recognition**: Track IDs denote spatio-temporal geometric continuity only. They do **not** represent global user identity, person recognition, biometric profiles, or database matches.

---

## 5. Output Video Visual Specifications (Step 5.12)

Annotated development videos render structured visual overlays:

```
┌──────────────────────┐
│ Track ID: 7          │
│ Confidence: 96%      │
│ Quality: 91/100      │
│ Status: Excellent    │
└──────────────────────┘
    •      •
       •
     •   •
```

* **Bounding Box**: Rendered in green (or status-coded color).
* **Track Card**: Solid dark badge rendering `Track ID: X`, `Confidence: XX%`, optional `Quality: XX/100`, and `Status`.
* **Facial Landmarks**: 5 canonical points (eyes, nose, mouth corners) in distinct color palette.
* **Persistent Session HUD**: Overlaid top-left: `Frame: X/N | Active: Y | Peak: Z | Unique: U`.
* **Prohibition**: **Zero identity names or biometric identifiers are ever rendered.**

---

## 6. Tracking & Telemetry Metrics (Step 5.13)

The pipeline records actual measured metrics:

```json
{
  "total_frames": 300,
  "processed_frames": 300,
  "total_processing_time": 4580.2,
  "average_frame_processing_time": 15.27,
  "average_fps": 65.5,
  "peak_simultaneous_faces": 3,
  "unique_track_count": 4,
  "track_lost_count": 2,
  "total_faces_detected": 892,
  "detection_time_ms": 3120.5,
  "tracking_time_ms": 140.2,
  "analysis_time_ms": 520.1,
  "annotation_time_ms": 799.4
}
```

*Note on Evaluation*: DeepVision AI reports actual measured system execution throughput and trajectory statistics. Tracking accuracy metrics (e.g. MOTA, IDF1, HOTA) are not claimed without evaluation on a certified, ground-truth labeled benchmark dataset.

---

## 7. Asynchronous Job Lifecycle & Storage (Step 5.16, 5.19, 5.21)

Video processing tasks are handled via an asynchronous job abstraction decoupled from the HTTP request-response cycle:

```
[POST /api/v1/video/analyze]
            ↓
       Job Created
       (status: queued)
            ↓
  Worker Thread Pool Launches
       (status: processing)
    ┌───────┴───────┐
    ▼               ▼
Completed        Cancelled / Failed
(status: completed) (status: cancelled / failed)
[GET /jobs/{id}/download]
```

### Job States
- `queued`: Video uploaded, validated, and enqueued for worker execution.
- `processing`: Frame streaming loop actively running face detection, ByteTrack tracking, analysis, and annotation. Progress percentage is dynamically updated (`0%` to `100%`).
- `completed`: Pipeline finished; metrics telemetry compiled and annotated video written to storage. Client can stream output via `/api/v1/video/jobs/{job_id}/download`.
- `cancelled`: Client issued `POST /api/v1/video/jobs/{job_id}/cancel`. Worker safely halted, video writers flushed and closed, and partial artifacts purged.
- `failed`: Processing encountered an error (e.g. corrupt codec or disk full). Safe, user-facing error message recorded; internal traces are kept server-side.

### Storage Abstraction
File storage is mediated by `LocalStorageBackend` implementing `BaseStorageBackend`. Disk paths are strictly isolated on the host system and never exposed across the REST interface.

---

## 8. Resource Limits & Security Bounds (Step 5.24)

To prevent resource exhaustion and Denial-of-Service attacks, the following limits are enforced:

| Constraint | Limit | Description |
| :--- | :---: | :--- |
| `VIDEO_MAX_FILE_SIZE_MB` | `500 MB` | Upper threshold for uploaded video containers. |
| `VIDEO_MAX_DURATION_SECONDS` | `600 s` (10 min) | Maximum video presentation duration. |
| `VIDEO_MAX_FRAME_COUNT` | `18,000 frames` | Hard limit on total decodable frames. |
| `VIDEO_MAX_WIDTH` | `3840 px` (4K) | Maximum allowable video frame width. |
| `VIDEO_MAX_HEIGHT` | `2160 px` (4K) | Maximum allowable video frame height. |
| Supported Containers | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` | Validated by magic byte probing and stream header inspection. |

---

## 9. Known Tracking Limitations

* **Severe Extended Occlusions**: When a face is occluded for longer than `max_time_lost` frames, the track is terminated. Upon re-emergence, a new track ID will be assigned.
* **Extreme Cross-Overs**: When two faces of identical scale directly overlap and cross trajectories at high speed, geometric IoU matching may occasionally suffer track switching without appearance embeddings.
* **Violent Camera Motion**: In handheld footage with sudden jarring camera pans, visual motion blur can cause temporary detection drops; the Kalman filter coasts the bounding box, but extreme shifts may require track re-initialization.
