# Facial Landmark Strategy & Model Selection Report

**Project**: DeepVision AI  
**Document**: Landmark Strategy & Architectural Evaluation (Step 4)  
**Target Runtimes**: CPU (Native SIMD), ONNX Runtime, OpenCV DNN  
**Status**: Completed & Approved  

---

## 1. Executive Summary & Strategy Determination

In Step 4, we must establish a facial landmark strategy for quality assessment, facial orientation, and head pose estimation.

### Key Determination:
**The active DeepVision face detector (`YuNet` / `SCRFD` ONNX) already predicts 5-point facial landmarks natively in the initial detection forward pass.**

Every detection produces:
1. `right_eye` $(x, y)$
2. `left_eye` $(x, y)$
3. `nose_tip` $(x, y)$
4. `mouth_right` $(x, y)$
5. `mouth_left` $(x, y)$

### Strategic Decision:
**We will USE the native 5-point facial landmarks produced directly by the existing detector.**

#### Engineering Rationale:
* **Zero Additional Latency**: Landmark regression heads are evaluated concurrently with bounding box regression inside the detector's single forward pass. No second-stage cropped ROI forward pass is required.
* **Zero Extra Dependencies**: Eliminates the need to install or bundle heavyweight secondary frameworks (such as Google MediaPipe or InsightFace PyTorch modules) that add binary bloat or complex wheel build requirements on Windows/Linux.
* **Fully Sufficient for Step 4 Requirements**: The 5 canonical facial keypoints are mathematically sufficient to calculate:
  - **Inter-Pupillary Distance (IPD)** for face scale analysis.
  - **Facial symmetry and geometry ratios** (eye-to-nose, eye-to-mouth).
  - **3D Head Pose Estimation** (Yaw, Pitch, Roll) and frontality scoring using Perspective-n-Point (`cv2.solvePnP`) with a 3D canonical facial model.
* **Universal Portability & Permissive Licensing**: Operates natively in OpenCV C++ and ONNX Runtime under permissive **MIT License**, free of copyleft or non-commercial restrictions.

---

## 2. Comparative Evaluation Matrix

Although the detector provides native 5-point landmarks, we evaluated whether integrating a secondary standalone landmark model (e.g. 68-point or 468-point mesh) is justified for production:

| Evaluation Dimension | **YuNet Native (Selected)** | **MediaPipe Face Landmarker** | **InsightFace 2D-106 (ONNX)** | **PIPNet / 68-Point ONNX** |
| :--- | :---: | :---: | :---: | :---: |
| **Landmark Count** | **5 Keypoints** | 468 / 478 Mesh Points | 106 Points | 68 Points |
| **Pipeline Architecture** | **1-Stage (Integrated)** | 2-Stage (Detector + Crop Net) | 2-Stage (Detector + Crop Net) | 2-Stage (Detector + Crop Net) |
| **Additional Forward Passes**| **0 (Included in detector)**| +1 pass per detected face | +1 pass per detected face | +1 pass per detected face |
| **Additional CPU Latency** | **0.0 ms** (0 overhead) | ~5 – 12 ms per face | ~4 – 10 ms per face | ~6 – 15 ms per face |
| **Model Size** | **0 KB extra** (Part of 232 KB) | ~2.5 MB – 9.0 MB | ~5.0 MB – 12.0 MB | ~3.0 MB – 8.0 MB |
| **GPU / Provider Support** | CPU SIMD / CUDA / DirectML | WebAssembly, CPU, GPU | ONNX Runtime (CPU/CUDA) | ONNX Runtime (CPU/CUDA) |
| **Deployment Complexity** | **None (Already deployed)** | High (mediapipe C++ wheels) | Medium (ONNX crop pipeline) | Medium (crop pipeline) |
| **Python 3.12 Compatibility**| **100% Native** | Conditional / Version issues | Native via ONNX | Native via ONNX |
| **Software License** | **MIT License** | Apache 2.0 | Non-Commercial on weights* | MIT / Academic |
| **Maintenance Status** | Actively Maintained (OpenCV Zoo)| Actively Maintained (Google) | Maintained (DeepInsight) | Open Source Research |

*\*Note: Multiple InsightFace landmark model weights carry non-commercial research clauses that restrict commercial SaaS deployment.*

---

## 3. Technology Trade-Off Analysis

### A. YuNet Native 5-Point Landmarks (Selected Strategy)
* **How it works**: The neural network backbone branches into classification, box regression, and landmark regression heads. Landmark coordinates $(x_i, y_i)$ are regressed directly from feature anchors and inverted alongside bounding box coordinates.
* **Strengths**:
  - Requires **zero additional compute** or memory allocations.
  - Scales effortlessly to scenes with 10+ faces without multiplying latency by $N$ faces.
  - Perfect alignment with original image coordinate space.
* **Limitations**:
  - Does not provide dense eyebrow, eye contour, or lip perimeter points. (Dense contours are not required for Step 4 quality and pose estimation).

### B. MediaPipe Face Landmarker (Google)
* **How it works**: Uses a Blazeface detector followed by a dense 468-point 3D face mesh regression network.
* **Strengths**:
  - Extremely detailed 3D surface mesh suitable for AR filters, iris tracking, and facial animation.
* **Weaknesses**:
  - Requires a second forward pass for *every* detected face ($O(N)$ latency scaling).
  - Heavy dependency tree (`mediapipe`, Protobuf bindings) that frequently encounters wheel compatibility issues across disparate Python 3.11/3.12 operating system environments.
  - Excessive for basic image quality filtering and coarse head pose estimation.

### C. InsightFace 2D-106 Landmark Model
* **How it works**: Takes a cropped and aligned $112 \times 112$ or $192 \times 192$ face image and predicts 106 facial boundary keypoints.
* **Strengths**:
  - High geometric precision for face recognition alignment.
* **Weaknesses**:
  - Adds a secondary ONNX inference session and crop/affine transform preprocessing overhead.
  - Certain pre-trained InsightFace model checkpoints restrict commercial use.

---

## 4. Landmark Suitability for Step 4 Requirements

The native 5 keypoints directly fulfill all required Step 4 capabilities:

1. **Inter-Pupillary Distance (IPD)**:
   $$\text{IPD} = \sqrt{(x_{\text{left\_eye}} - x_{\text{right\_eye}})^2 + (y_{\text{left\_eye}} - y_{\text{right\_eye}})^2}$$
   Used to measure effective face resolution and biometric utility (e.g. ISO/IEC 19794-5 recommendation of minimum 30–60 pixels IPD).

2. **Facial Symmetry & Alignment**:
   - Compares eye-to-nose horizontal and vertical projections to detect partial occlusion and profile tilt.
   - Computes eye-line horizontal tilt angle:
     $$\theta_{\text{roll}} = \arctan2(y_{\text{left\_eye}} - y_{\text{right\_eye}}, x_{\text{left\_eye}} - x_{\text{right\_eye}})$$

3. **3D Head Pose Estimation (Perspective-n-Point)**:
   - Uses OpenCV's `cv2.solvePnP` with a standardized 3D anthropometric facial model (defined in camera coordinate space):
     - Right Eye: `(-33.0, 33.0, -25.0)`
     - Left Eye: `(33.0, 33.0, -25.0)`
     - Nose Tip: `(0.0, 0.0, 0.0)`
     - Mouth Right: `(-25.0, -38.0, -20.0)`
     - Mouth Left: `(25.0, -38.0, -20.0)`
   - Computes 3D Euler angles: **Yaw** (turning), **Pitch** (nodding), and **Roll** (tilting), providing robust frontality scoring without any secondary neural network.

---

## 5. Architectural Decision Summary

1. **Adopt Native 5-Point Landmarks**: No additional framework will be installed. The detection pipeline retains its single-pass efficiency.
2. **Modular Landmark Interface**: The analysis layer accepts any landmark sequence conforming to `List[LandmarkPoint]`, ensuring future drop-in support for dense meshes if 3D reconstruction is introduced in later milestones.
3. **Hardware Independence**: The landmark extraction and mathematical pose estimation run seamlessly on CPU with zero GPU dependencies.
