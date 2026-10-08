# DeepVision AI — AI Engine Architecture

This document describes the architectural design, component separation, and execution flow of the **DeepVision AI** inference engine.

---

## 1. Architectural Philosophy

DeepVision AI follows strict separation of concerns to ensure the AI subsystem is maintainable, swappable, and cloud-ready:

1. **Decoupled API and AI Subsystems**: Web routes never communicate directly with neural network weights or raw tensors. All interactions pass through the application-level `AIService`.
2. **Framework-Agnostic Interface**: The core detection contract is defined by `BaseFaceDetector`. Any future detector (SCRFD, YuNet, YOLO-Face, RetinaFace) can be plugged in without changing API controllers or business logic.
3. **Hardware Agility with Zero-Crash Fallback**: Hardware execution targets (`AUTO`, `CPU`, `GPU`) dynamically resolve at runtime. If GPU acceleration or vendor drivers are unavailable, the system automatically falls back to multi-threaded CPU execution without crashing.
4. **Strict Privacy and Memory Management**: Images are ingested into volatile memory, analyzed, and released. Raw image bytes, pixel matrices, and base64 buffers are **never** logged or saved to disk unless explicitly required.
5. **Zero Fabrication**: When model binaries are not configured or when an image contains no faces, the engine never produces fabricated or synthetic detections.

---

## 2. End-to-End Inference Flow

Every face detection request executes through a deterministic, hardened 9-stage processing flow:

```text
Image Upload
     ↓
Validation
     ↓
Preprocessing
     ↓
AI Inference
     ↓
Face Detection
     ↓
Confidence Filtering
     ↓
NMS / Postprocessing
     ↓
Bounding Box Validation
     ↓
Structured JSON Result
```

### Stage 1: Image Upload
* Ingests image data via HTTP `POST /api/v1/ai/detect` using `multipart/form-data` with form field `file`.
* Operates strictly in volatile memory (`io.BytesIO`) using stream chunking. No uploaded images are permanently stored on disk.

### Stage 2: Validation
* **MIME & Extension Whitelist**: Verifies declared MIME types (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`) and extensions (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
* **Binary Magic Numbers**: Inspects leading binary bytes to prevent file extension spoofing, polyglot files, and executable injection (`MZ`, `ELF`, `#!`, `<?php`).
* **Streaming Byte Cap**: Enforces hard file size limits (default 10 MB via `settings.AI_MAX_IMAGE_SIZE_MB`). Aborts immediately with `413 Payload Too Large`.
* **Structural Verification & Decompression Bomb Protection**: Verifies container tables with Pillow and enforces maximum dimensions ($4096 \times 4096\text{px}$ via `settings.AI_MAX_IMAGE_DIMENSION`) and pixel caps (100M pixels) to prevent pixel floods and memory exhaustion.
* **Filename Sanitization**: Cleans untrusted client-supplied filenames, stripping directory traversal sequences (`../`, `..\`) and null bytes.

### Stage 3: Preprocessing
* **Color Space Standardization**: Dynamically converts single-channel Grayscale, 4-channel RGBA (alpha blended over black), and BGR images into standardized 3-channel RGB NumPy uint8 arrays.
* **Aspect-Ratio Letterboxing**: Resizes the image to model input canvas ($640 \times 640\text{px}$) with symmetric padding, strictly preserving the original aspect ratio to avoid geometric distortion.

### Stage 4: AI Inference
* Evaluates input tensors using the active detection runtime:
  - **OpenCV FaceDetectorYN / YuNet C++ SIMD backend** for ultra-fast CPU inference (~15–25ms).
  - **ONNX Runtime session** supporting hardware execution providers (`CPUExecutionProvider`, `CUDAExecutionProvider`, `DmlExecutionProvider`).
* Automatic, zero-crash hardware fallback to CPU if GPU drivers are unavailable.

### Stage 5: Face Detection
* Multi-scale feature extraction across strides 8, 16, and 32 to detect faces across diverse scales (small far-field faces $< 20\text{px}$ up to close-up portraits).
* Raw proposal extraction: candidate bounding boxes, classification scores, and 5 facial landmark keypoints (right eye, left eye, nose tip, right mouth corner, left mouth corner).

### Stage 6: Confidence Filtering
* Gating proposals against configurable detection confidence threshold:
  - Global default configured via `AI_CONFIDENCE_THRESHOLD` (default: `0.6`).
  - Dynamically overridable on a per-request basis via query parameter `?confidence_threshold=0.75`.
* Proposals failing to meet the threshold are pruned prior to coordinate transformation.

### Stage 7: NMS / Postprocessing
* **Non-Maximum Suppression (NMS)**: Eliminates duplicate overlapping proposals using Intersection-over-Union (IoU) overlap thresholds (`AI_NMS_THRESHOLD`, default: `0.30`), preserving only the proposal with the strongest confidence score.
* **Coordinate Inversion (Unletterbox)**: Reverses preprocessing letterbox scaling and padding, translating normalized/resized coordinates back into the original image's native pixel dimensions.

### Stage 8: Bounding Box Validation
* Validates geometric integrity of each bounding box against original image bounds:
  - Ensures $x_1 \ge 0$, $y_1 \ge 0$, $x_2 \le \text{image\_width}$, and $y_2 \le \text{image\_height}$.
  - Ensures $x_2 > x_1$ and $y_2 > y_1$.
* Clips bounding boxes that partially overflow image boundaries (`clip_to_bounds`).
* Safely discards degenerate or out-of-bounds candidate detections.

### Stage 9: Structured JSON Result
* Formats detection outputs into strongly-typed Pydantic v2 schemas (`AIDetectionResponse`):
  - Returns `model`, `device`, `image` dimensions, `face_count`, `faces` array (with `bbox`, `confidence`, `landmarks`), and granular `timing` breakdown.
  - Generates request tracing header `X-Request-ID`.
  - Records lightweight, privacy-preserving structured telemetry logs (zero raw image bytes logged).

---

## 3. Directory Layout

The AI engine code is isolated under `backend/app/ai/`:

```text
backend/app/ai/
├── __init__.py
├── detection/               # Face detection domain
│   ├── __init__.py
│   ├── base.py              # BaseFaceDetector abstract interface
│   ├── detector.py          # FaceDetector implementation (YuNet & ONNX)
│   ├── schemas.py           # Re-exported detection schemas
│   ├── preprocessing.py     # ImagePreprocessor & letterboxing
│   └── adapter.py           # Model adapters & coordinate inversion
│
├── tracking/                # Multi-face trajectory tracking (Future Step)
│   └── __init__.py
│
├── analysis/                # Face quality & landmark analytics (Step 4)
│   ├── __init__.py          # Exported analysis modules and schemas
│   ├── schemas.py           # Pydantic v2 analysis data contracts
│   ├── landmarks.py         # 5-point landmark metrics (IPD, tilt, symmetry)
│   ├── pose.py              # 3D Head pose (Yaw, Pitch, Roll via solvePnP)
│   ├── metrics.py           # Blur/clarity, illumination, face size metrics
│   └── quality.py           # Composite quality scoring & FaceAnalyzer
│
├── recognition/             # Facial embedding & verification contract (Future Step)
│   ├── __init__.py
│   └── base.py              # BaseIdentityVerifier interface
│
├── visualization/           # Development & offline visualization tools
│   ├── __init__.py
│   └── annotator.py         # DetectionVisualizer (BBoxes, confidence, landmarks)
│
├── inference/               # Hardware runtime abstraction
│   ├── __init__.py
│   ├── device.py            # Device resolution (AUTO, CPU, GPU fallback)
│   ├── runtime.py           # ONNX Runtime session manager
│   └── onnx_engine.py       # Inference engine session
│
└── common/                  # Shared CV types and aliases
    ├── __init__.py
    └── types.py             # ImageSource, DeviceType, Coordinate definitions
```

---

## 4. Error Handling and Shielding

All AI subsystem exceptions inherit from `AIException` and map to HTTP status codes via centralized handlers:

| Exception | HTTP Status | Client Response | Server-Side Log |
|---|:---:|---|---|
| `ModelLoadError` | `503` | Sanitized error JSON | Full traceback with file/line |
| `DetectorNotConfiguredException` | `503` | Development setup instructions | Warning with configured path |
| `InferenceError` | `500` | Generic inference error | Full traceback with tensor shapes |
| `InvalidImageError` | `400` | Human-readable explanation | Warning with request metadata |
| `CorruptedImageException`| `400` | Corrupted byte notice | Warning with decoding error |
| `EmptyImageException` | `400` | 0-byte file notice | Warning |
| `ImageTooLargeException` | `413` | Payload size exceeded notice | Warning with byte counts |
| `UnsupportedImageError` | `400` | Allowed formats guidance | Warning with uploaded MIME type |
| `DeviceNotAvailableError`| `503` | Device unavailable notice | Detailed driver diagnostics |

Clients never receive internal Python tracebacks, memory pointers, or local filesystem paths.

---

## 5. Face Analysis Architecture (Step 4)

In addition to detection, DeepVision AI incorporates a dedicated, modular Face Analysis layer built on top of the face detector:

```text
Image
 ↓
Face Detector
 ↓
Face Bounding Box
 ↓
Face Crop
 ↓
Facial Landmarks (5-point)
 ↓
Quality & Pose Analysis
 ↓
Structured Face Analysis
```

### Analytical Capabilities:
1. **Facial Landmarks & Geometric Metrics**:
   - 5 canonical facial landmarks: right eye, left eye, nose tip, mouth right corner, mouth left corner.
   - Inter-pupillary distance (IPD) in image pixels.
   - Eye tilt angle ($\theta^\circ$) from horizontal.
   - Facial bilateral symmetry ratio ($0.0 \dots 1.0$).
2. **3D Head Pose & Orientation Estimation**:
   - Solves Perspective-n-Point (`cv2.solvePnP` with EPNP / iterative fallback) against a canonical 3D anthropometric face model.
   - Estimates Euler angles: **Yaw**, **Pitch**, and **Roll** in degrees ($[-90^\circ, +90^\circ]$).
   - Computes normalized **Frontality Score** ($0.0 \dots 1.0$) and qualitative orientation categorization (`FRONTAL`, `SLIGHT_TURN`, `PROFILE`, `UPWARD`, `DOWNWARD`).
3. **Clarity & Blur Detection**:
   - Measures high-frequency spatial variation using Laplacian variance.
   - Categorizes sharpness into `SHARP`, `SLIGHTLY_BLURRED`, `BLURRED`, `VERY_BLURRED`.
4. **Illumination & Exposure Diagnostics**:
   - Measures mean luminance brightness ($0 \dots 255$) and standard deviation contrast.
   - Categorizes exposure into `OPTIMAL`, `UNDEREXPOSED`, `OVEREXPOSED`, `LOW_CONTRAST`, `HARSH_CONTRAST`.
5. **Face Size & Resolution Analysis**:
   - Evaluates face bounding box width, height, aspect ratio, area, and canvas coverage ratio.
   - Classifies face scale into `TINY` ($<32\text{px}$), `SMALL` ($32\text{--}80\text{px}$), `MEDIUM` ($80\text{--}200\text{px}$), and `LARGE` ($>200\text{px}$).
   - Checks biometric processing sufficiency threshold.
6. **Composite Quality Scoring & Usability**:
   - Weighted multi-metric quality score combining clarity ($40\%$), illumination ($30\%$), resolution scale ($15\%$), and head frontality ($15\%$).
   - Qualitative grading: `EXCELLENT`, `GOOD`, `FAIR`, `POOR`, `UNUSABLE`.
   - Actionable binary `is_usable` verdict for biometric and verification downstream workflows.
7. **Dedicated API Endpoint**:
   - `POST /api/v1/ai/analyze` returning strongly typed `AIAnalysisResponse` with complete `StructuredFaceAnalysis` per face.

---

## 6. Current Capabilities (End of Step 4)

At the completion of Step 4, the DeepVision AI platform provides:

* **Real Image Face Detection**: Production YuNet ONNX inference on single/multi-face scenes.
* **5-Point Facial Landmarks**: Natively parsed during detection pass with zero overhead.
* **Facial Geometry Diagnostics**: IPD, eye-line tilt angle, and bilateral facial symmetry.
* **3D Head Pose Estimation**: Perspective-n-Point yaw, pitch, roll, frontality score, and pose classification.
* **Clarity & Blur Detection**: Laplacian variance sharpness, blur score, and categorical rating.
* **Illumination Diagnostics**: Luminance brightness, standard deviation contrast, and exposure classification.
* **Face Scale & Resolution Diagnostics**: Width, height, aspect ratio, canvas area fraction, and size categories.
* **Composite Quality Scoring**: Explainable, weighted quality score ($0.0 \dots 1.0$), qualitative grades, and usability verdict.
* **Backward Compatible Quality API**: Legacy `FaceQualityScore` preserved for existing callers.
* **REST API Endpoints**: `POST /api/v1/ai/detect` for fast face detection, and `POST /api/v1/ai/analyze` for comprehensive face analysis.

---

## 7. What is NOT Implemented Yet

To preserve architectural boundaries and development discipline, the following features are explicitly **NOT** implemented in Step 4:

* ❌ **Identity Recognition / Biometric Matching**: No facial embeddings extraction, identity database matching, or biometric identification. All detections maintain `face_id: null`.
* ❌ **Multi-Frame Video Tracking**: No temporal trajectory tracking across video frames (e.g. ByteTrack, SORT) or track ID continuity.
* ❌ **Webcam Streaming WebSocket**: Real-time live webcam frame-by-frame streaming over WebSocket is not yet connected to the deep face detection engine.
* ❌ **Liveness & Anti-Spoofing Detection**: No 2D/3D print attack, screen replay, or silicon mask anti-spoofing verification models.
* ❌ **Final Production Dashboard**: The analytics UI dashboard is not built yet.


