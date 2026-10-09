# Multi-Face Tracking Specification

**DeepVision AI — Multi-Face Tracking Engine Documentation**  
**Algorithm**: ByteTrack (Two-Stage Kinematic Data Association)  
**Author**: DeepVision AI Computer Vision Engineering Team  
**Status**: ACTIVE  
**Milestone**: Step 5 (Multi-Face Tracking)

---

## 1. Tracker Selection & Rationale

DeepVision AI utilizes **ByteTrack** for video multi-face tracking.

### Algorithm Comparison Summary

| MOT Algorithm | Association Logic | Compute Requirement | Biometric Embedding | Suitability for Face MOT |
| :--- | :--- | :--- | :--- | :--- |
| **ByteTrack** (Selected) | Two-stage IoU / Kalman kinematics | **CPU-optimized** (~2 ms/frame) | **None** (pure geometry) | **Ideal**: high FPS, low footprint, zero identity profiles |
| **SORT** | Single-stage IoU / Kalman | CPU (~1 ms/frame) | None | High ID switching during partial occlusions |
| **DeepSORT** | Deep Re-ID CNN + IoU | Heavy CPU/GPU (~35 ms/frame) | Yes (128-dim facial vectors) | Violates ephemeral tracking rules; high compute overhead |
| **BoT-SORT** | Camera motion compensation + Re-ID | GPU required (~45 ms/frame) | Yes | Excessive latency for production CPU deployment |

### Selection Rationale
1. **Zero Biometric Identity Profiling**: ByteTrack operates strictly on spatial bounding-box kinematics ($[x, y, a, h]$) and velocities. It does **not** extract or store facial recognition vectors.
2. **Two-Stage Association (BYTE Scheme)**: Low-confidence detections ($0.15 \le \text{conf} < 0.50$) caused by motion blur, occlusions, or head turns are recovered in the second association phase, drastically reducing fragmented tracks.
3. **Ultra-Low Latency on CPU**: Averages $\sim 2.2\text{ ms}$ tracking latency per frame on standard x86 CPU hardware, preserving system resources for face detection and analytics.

---

## 2. Track ID Semantics & Identity Rules

In strict compliance with privacy and security mandates:
* **Ephemeral Scope**: Track IDs (`track_id: 1, 2, 3...`) are valid **only for the duration of a single video processing job**.
* **Sequential Integers**: Each new verified track increments an internal counter starting from `1`.
* **No Facial Recognition**: A Track ID indicates:
  > *"This detection appears to belong to the same spatial visual trajectory across nearby video frames."*
  
  It does **not** signify:
  > *"This is Person X or a known identity."*
* **Session Reset**: Between jobs, `tracker.reset()` flushes all Kalman states, track caches, and resets the ID counter.

---

## 3. Algorithm Architecture & Mathematical Mechanics

### A. State Space Representation
Each face trajectory is modeled with an 8-dimensional state vector $\mathbf{x}$:

$$\mathbf{x} = [x_c, y_c, a, h, \dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]^T$$

where:
* $(x_c, y_c)$ represents bounding box center coordinates.
* $a = \frac{w}{h}$ represents aspect ratio.
* $h$ represents bounding box height.
* $[\dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]$ represent corresponding velocities under a continuous constant-velocity kinematic model.

### B. Two-Stage BYTE Association
For each frame $k$:
1. **Partition Detections**: Detections are divided into high-confidence ($\mathcal{D}_{\text{high}}$, $\text{conf} \ge \tau_{\text{high}}$) and low-confidence ($\mathcal{D}_{\text{low}}$, $\tau_{\text{low}} \le \text{conf} < \tau_{\text{high}}$).
2. **Stage 1 Association**:
   - Predict next positions for all active tracks $\mathcal{T}_{\text{active}}$ using the Kalman filter.
   - Compute spatial Intersection-over-Union (IoU) cost matrix against $\mathcal{D}_{\text{high}}$.
   - Solve linear sum assignment with IoU threshold $\tau_{\text{match}} = 0.5$.
   - Matched tracks update their Kalman filters.
3. **Stage 2 Association**:
   - Remaining unmatched tracks from Stage 1 are matched against $\mathcal{D}_{\text{low}}$.
   - This recovers faces that experienced sudden brightness changes, head turns, or partial occlusions without terminating the track.
4. **Track Lifecycle Transitions**:
   - Unmatched high-confidence detections spawn **Tentative** tracks.
   - After `min_hits` consecutive detections (default: 2), tentative tracks become **Confirmed**.
   - Confirmed tracks that remain unmatched transition to **Lost**.
   - If a track remains lost for more than `max_time_lost` frames (default: 30), it is **Removed**.

---

## 4. Processing Metrics & Telemetry

The tracking engine produces structured telemetry per job:
* `total_frames`: Total frames in video container.
* `processed_frames`: Total frames evaluated.
* `total_processing_time`: Wall-clock pipeline execution time (ms).
* `average_fps`: Realized processing throughput.
* `peak_simultaneous_faces`: Maximum faces tracked concurrently in any single frame.
* `unique_track_count`: Total unique Track IDs allocated throughout the video.
* `track_lost_count`: Total track disappearance events.
* `detection_time_ms`, `tracking_time_ms`, `analysis_time_ms`, `annotation_time_ms`: Granular pipeline timing.

---

## 5. Resource Limits & Configuration

Configurable via `app/core/config.py`:
* `VIDEO_DETECTION_INTERVAL` (default: `1`): Face detector run frequency.
* `VIDEO_ANALYSIS_INTERVAL` (default: `5`): Deep face diagnostics run frequency.
* `VIDEO_MAX_FILE_SIZE_MB` (default: `500 MB`).
* `VIDEO_MAX_DURATION_SECONDS` (default: `600 s`).
* `VIDEO_MAX_FRAME_COUNT` (default: `18000`).

---

## 6. Known Limitations

1. **Occlusion Exceeding Max Window**: If a face is obstructed for $> 30$ frames (1 second at 30 fps), its track is retired. Upon reappearance, a new Track ID will be generated.
2. **Co-linear Crossing Trajectories**: When two subjects pass directly in front of each other with identical bounding box dimensions, spatial IoU matching may swap tracks in the absence of visual appearance features.
3. **Severe Camera Jitter**: Extreme camera shakes can exceed the linear constant-velocity Kalman prediction bounds.
