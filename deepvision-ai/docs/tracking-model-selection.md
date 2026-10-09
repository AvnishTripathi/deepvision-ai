# Multi-Object Tracking (MOT) Algorithm Selection & Architectural Trade-off Analysis

**DeepVision AI — Video Face Detection & Multi-Face Tracking Engine**  
**Author**: DeepVision AI Computer Vision Engineering Team  
**Status**: APPROVED  
**Target Milestone**: Step 5 (Video Processing & Multi-Face Tracking)  

---

## 1. Executive Summary

This document evaluates Multi-Object Tracking (MOT) algorithms for the **DeepVision AI** video processing subsystem. The tracking engine must track multiple human faces continuously across video frames, maintain persistent temporary track identifiers (`track_id`), operate deterministically on commodity CPU infrastructure, and handle brief occlusions and scale variations without relying on biometric facial feature vectors or re-identification databases.

After rigorous comparative analysis against operational criteria, **ByteTrack** has been selected as the initial multi-face tracking implementation for DeepVision AI.

---

## 2. Tracking Paradigm & Problem Formulation for Faces

Tracking human faces in video poses distinct challenges compared to standard generic object tracking (e.g., vehicles, full human bodies):

1. **Abrupt Confidence Fluctuations**: Human faces frequently exhibit sharp dips in detector confidence during rapid head rotations (yaw/pitch excursions), motion blur, and expressions. Standard trackers that apply a single rigid detection confidence cutoff drop these detections, causing broken tracks and frequent ID switching.
2. **Partial Occlusions**: Faces are frequently subject to transient occlusions by hands, drinking cups, microphones, eyeglasses, and passing obstacles.
3. **Strict Privacy & Non-Identification Invariants**: As established in Step 22 and [`docs/security-and-privacy.md`](./security-and-privacy.md), DeepVision AI strictly prohibits:
   - Creating persistent biometric templates or embedding vectors.
   - Performing facial recognition or person re-identification across independent sessions or against historical databases.
   - Track IDs must represent **temporary, video-session-scoped geometric kinematic tracks only**.
4. **Computational Budget**: Face tracking must execute synchronously or near-real-time on CPU alongside frame decoding and YuNet inference, without introducing heavyweight secondary deep neural network passes.

---

## 3. Algorithm Candidates Evaluated

We evaluated five candidate multi-object tracking architectures:

1. **SORT (Simple Online and Realtime Tracking)** (Bewley et al., 2016)
2. **ByteTrack (Multi-Object Tracking by Associating Every Detection Box)** (Zhang et al., ECCV 2022)
3. **DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)** (Wojke et al., ICIP 2017)
4. **BoT-SORT (Robust Associations Multi-Pedestrian Tracker)** (Aharon et al., 2022)
5. **Centroid / Euclidean Distance Tracking** (Baseline Heuristic)

---

## 4. Comprehensive Comparative Evaluation

| Evaluation Criteria | SORT | ByteTrack | DeepSORT | BoT-SORT | Centroid Tracker |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tracking Stability** | Moderate | **High** | High | High | Low |
| **CPU Performance** | Ultra-Fast ($>1,000$ fps) | **Ultra-Fast ($>800$ fps)** | Slow ($15–40$ fps) | Moderate ($60–120$ fps) | Ultra-Fast ($>2,000$ fps) |
| **GPU Requirements** | None (CPU only) | **None (CPU only)** | Recommended / Required | Optional (GMC adds CPU load) | None |
| **Implementation Complexity** | Low (~150 lines) | **Moderate (~350 lines)** | High (~1,200 lines) | High (~1,500 lines) | Very Low (~100 lines) |
| **External Dependencies** | `numpy`, `scipy` | **`numpy`, `scipy`** | PyTorch / ONNX + Re-ID weights | OpenCV, PyTorch / Cython | `numpy`, `scipy` |
| **Suitability for Faces** | Good | **Exceptional** | Problematic (Re-ID) | Overkill (camera motion) | Poor (fails on scale changes) |
| **Occlusion Handling** | Limited (deletes on missed frames) | **Superior (low-score matching)** | Good (visual appearance) | High | Minimal |
| **ID Switching Rate** | High during confidence dips | **Low** | Low | Low | Very High |
| **Deployment Footprint** | $< 50\text{ KB}$ | **$< 100\text{ KB}$** | $25–80\text{ MB}$ (model weights) | $30–100\text{ MB}$ | $< 20\text{ KB}$ |
| **Privacy Policy Compliance** | Compliant (pure kinematics) | **Compliant (pure kinematics)** | **Non-Compliant** (extracts Re-ID embeddings) | Partial (Re-ID mode conflicts) | Compliant |
| **Open Source Licensing** | BSD-2-Clause / GPL | **MIT License** | GPL-3.0 / MIT variants | MIT License | Public Domain / MIT |
| **Maintenance Status** | Academic baseline | **Actively maintained & industry standard** | Legacy / Superseded | Actively maintained | Basic reference |

---

## 5. Detailed Analysis of Evaluated Options

### 5.1 SORT (Simple Online and Realtime Tracking)
* **Mechanism**: Combines a standard constant-velocity 2D Kalman Filter with the Hungarian algorithm (linear sum assignment) using bounding box Intersection-over-Union (IoU) as the association cost.
* **Strengths**: Minimal mathematical overhead, zero deep learning dependencies, executes in microseconds per frame.
* **Weaknesses**: Discards all detections below the primary detector threshold. When a face turns sideways or blurs, confidence drops below the threshold; SORT fails to associate, loses tracklet continuity, and creates a new track ID when the face is re-detected, causing excessive ID switches.

### 5.2 ByteTrack (Selected Architecture)
* **Mechanism**: Introduces a two-stage association strategy based on BYTE:
  1. Detections are partitioned into high-confidence ($D_{\text{high}}$) and low-confidence ($D_{\text{low}}$) sets.
  2. First association matches active tracklets with $D_{\text{high}}$ using Kalman Filter state prediction and IoU distance.
  3. Second association matches remaining unmatched tracklets with $D_{\text{low}}$ using IoU distance.
  4. Unmatched high-confidence detections initialize new tentative tracklets. Unmatched tracklets persist in a lost state for up to $T_{\text{lost}}$ frames before eviction.
* **Why it Excels for Face Tracking**:
  - Head turns, motion blur, and partial hand occlusions routinely degrade detection confidence to $0.20–0.50$. Under ByteTrack, these low-confidence detections are not discarded as background noise; instead, they successfully maintain continuity of existing tracks without introducing false-positive track creations.
  - Operates purely on bounding box geometries and Kalman kinematics without needing deep neural feature extractors.
  - Zero GPU requirement; CPU execution is orders of magnitude faster than video frame decoding.

### 5.3 DeepSORT
* **Mechanism**: Augments SORT with a deep convolutional neural network trained on metric learning to generate appearance feature vectors (embeddings), computing a fused cosine distance and Mahalanobis distance metric.
* **Disqualification Reasons**:
  1. **Privacy Invariant Conflict**: Generating face appearance embeddings introduces biometric vector extraction into the video tracking pipeline, directly violating Step 22 and DeepVision AI's non-identification privacy policy.
  2. **Severe CPU Bottleneck**: Extracting appearance vectors for every detected face crop requires multiple CNN forward passes per frame, degrading CPU throughput to $10–25\text{ fps}$.
  3. **Operational Complexity**: Requires managing separate Re-ID model weights, image resizing pipelines, and cosine metric memory banks.

### 5.4 BoT-SORT
* **Mechanism**: Incorporates Global Motion Compensation (GMC) via camera background optical flow or ORB feature matching, alongside modified Kalman Filter state representations $(x, y, w, h)$.
* **Disqualification Reasons**:
  - Global Motion Compensation is designed for moving cameras (e.g., drones, body cameras, handheld sports capture). In typical video analytics scenarios, GMC adds substantial CPU overhead with diminishing returns for facial tracking.

---

## 6. Selected Decision & Architecture

### Decision: ByteTrack (Kinematic Multi-Stage IoU Tracking)

DeepVision AI adopts **ByteTrack** as the core multi-face tracking engine, customized specifically for face geometry and the OpenCV YuNet detection pipeline:

1. **Detector Decoupling**: Reuses the existing `FaceDetector` (YuNet) without modification. YuNet outputs high-accuracy bounding boxes and detection confidence scores that feed directly into ByteTrack's two-stage association.
2. **Kalman Motion Model**: 8-dimensional state vector:
   $$x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}, \dot{r}]^T$$
   where $(u, v)$ is the bounding box center, $s$ is the bounding box area ($w \times h$), $r$ is the aspect ratio ($w / h$), and their corresponding first-order velocities.
3. **Track Lifecycle State Machine**:
   - `Tentative`: Newly detected face requiring $N_{\text{init}}$ consecutive hits (default: 3 frames) to be confirmed.
   - `Confirmed`: Active, confirmed track assigned a persistent temporary `track_id`.
   - `Lost`: Missed in current frame, coasted using Kalman prediction for up to $T_{\text{max\_age}}$ frames (default: 30 frames).
   - `Deleted`: Permanently removed after exceeding $T_{\text{max\_age}}$ missed frames.
4. **Scoped Ephemeral Identity**: Track IDs (`track_id: 1, 2, ...`) are strictly sequential integers generated per video processing job and reset immediately upon job completion.

---

## 7. Hyperparameter Configuration Guidelines

| Parameter | Recommended Default | Description |
| :--- | :---: | :--- |
| `high_detection_threshold` | `0.50` | Primary detection threshold separating high vs. low confidence detections. |
| `low_detection_threshold` | `0.15` | Minimum floor threshold for secondary association detections. |
| `match_threshold_first` | `0.70` | Maximum cost (1 - IoU) for high-confidence association. |
| `match_threshold_second` | `0.50` | Maximum cost (1 - IoU) for secondary low-confidence association. |
| `max_time_lost` | `30` frames | Maximum frames to retain lost tracks before permanent deletion (~1 second at 30 fps). |
| `min_hits` | `3` frames | Minimum consecutive frame hits required to promote tentative track to confirmed. |

---

## 8. Summary of Alignment with Project Constraints

* **Reuses existing face detector**: 100% compliant (consumes `FaceDetector.detect()` outputs).
* **No secondary detector**: 100% compliant.
* **No biometric profiling / face recognition**: 100% compliant (no appearance embeddings).
* **Zero GPU requirements**: 100% compliant (executes entirely on CPU via NumPy / SciPy).
* **Deterministic track persistence**: 100% compliant (coasts through brief face occlusions and head turns).
