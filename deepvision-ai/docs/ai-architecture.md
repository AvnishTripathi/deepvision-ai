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

Every detection request travels through a deterministic 7-stage processing pipeline:

```text
       ┌────────────────────────┐
       │     Incoming Image     │  (Multipart Upload / Bytes)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   1. File Validation   │  (MIME type, size limit, dimensions)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │    2. Preprocessing    │  (RGB decode, letterbox, NCHW tensor)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  3. Inference Runtime  │  (ONNX Runtime session, CPU/GPU)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │    4. Face Detector    │  (Anchor decoding, strides, NMS, landmarks)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ 5. Structured Results  │  (BoundingBox, FaceDetection, telemetry)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │    6. API Response     │  (Sanitized JSON, X-Request-ID, HTTP status)
       └────────────────────────┘
```

### Stage 1: Validation
* **File Type Check**: Ensures content is a supported image format (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`). Rejects text, executables, or PDFs with `400 Bad Request` (`unsupported_media_type`).
* **Payload Size Check**: Enforces memory safety against denial-of-service attempts. Ingests bytes in-memory and validates that payload size $\le \text{AI\_MAX\_IMAGE\_SIZE\_MB}$ (default 10 MB). Rejects oversized payloads with `413 Content Too Large` (`payload_too_large`).
* **Integrity Check**: Rejects empty `0-byte` buffers and corrupt streams that fail image decoding with `400 Bad Request` (`empty_file`, `corrupted_image`).

### Stage 2: Preprocessing
* **Color Standardization**: Converts BGR, Grayscale, or RGBA inputs into standardized uint8 RGB NumPy arrays.
* **Aspect-Ratio Letterboxing**: Resizes the image to the model input dimensions (e.g. $640 \times 640$) while preserving the native aspect ratio through symmetric padding. Prevents aspect distortion that harms detection accuracy.
* **Tensor Normalization**: Normalizes pixel values to $[0.0, 1.0]$ float32 and permutes dimensions to the standard NCHW tensor layout $(1, 3, 640, 640)$.

### Stage 3: Inference Runtime
* **Provider Orchestration**: Manages the underlying ONNX Runtime session using `CPUExecutionProvider` or accelerated GPU providers (`CUDAExecutionProvider`, `TensorrtExecutionProvider`, `DmlExecutionProvider`).
* **Thread Optimization**: Allocates worker threads according to host CPU core topology to maximize throughput while avoiding thread contention.
* **Session Warmup**: Executes graph optimization passes on startup to avoid cold-start latency on initial requests.

### Stage 4: Face Detector & Architecture Adapter
* **Forward-Pass Evaluation**: Evaluates input tensors through the neural network.
* **Multi-Scale Anchor Decoding**: For SCRFD, processes outputs across strides 8, 16, and 32 to detect faces ranging from small far-field faces ($< 16\text{px}$) to large foreground portraits.
* **Non-Maximum Suppression (NMS)**: Eliminates redundant overlapping bounding boxes based on Intersection-over-Union (IoU) thresholds.
* **Landmark Alignment**: Decodes 5 facial keypoint landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner).
* **Unletterbox Coordinate Inversion**: Inverts preprocessing letterbox scaling and padding, translating box and landmark coordinates back into original image pixel space.

### Stage 5: Structured Detection Result
* Normalizes detection objects into strongly-typed Pydantic v2 schemas:
  - `BoundingBox`: Absolute coordinates (`x1`, `y1`, `x2`, `y2`), dimensions, and area.
  - `NormalizedBoundingBox`: Relative coordinates in $[0.0, 1.0]$ image space.
  - `FaceDetection`: Identity, bounding box, confidence score, landmarks, and face quality metrics.
  - `FrameInferenceResult`: Latency breakdown (`preprocessing_time_ms`, `inference_time_ms`, `postprocessing_time_ms`, `total_time_ms`), device used, and model name.

### Stage 6: API Response & Structured Telemetry
* The endpoint formats the result into a standardized JSON response:
  ```json
  {
    "success": true,
    "model": "scrfd_500m",
    "device": "CPU",
    "image": {
      "width": 1920,
      "height": 1080
    },
    "processing_time_ms": 28.5,
    "faces": [
      {
        "bbox": {
          "x1": 100.0,
          "y1": 120.0,
          "x2": 300.0,
          "y2": 350.0
        },
        "confidence": 0.96
      }
    ],
    "request_id": "8f3b2d1e-4c7a-4a2b-9e3d-1a2b3c4d5e6f"
  }
  ```
* Emits a privacy-preserving structured telemetry log record (capturing model, device, resolution, face count, latency, and request ID).

---

## 3. Directory Layout

The AI engine code is isolated under `backend/app/ai/`:

```text
backend/app/ai/
├── __init__.py
├── detection/               # Face detection domain
│   ├── __init__.py
│   ├── base.py              # BaseFaceDetector abstract interface
│   ├── detector.py          # FaceDetector implementation
│   ├── schemas.py           # Re-exported detection schemas
│   ├── preprocessing.py     # ImagePreprocessor & letterboxing
│   └── adapter.py           # SCRFDAdapter & BaselineAdapter
│
├── tracking/                # Multi-face trajectory tracking (Step 3)
│   └── __init__.py
│
├── analysis/                # Facial analytics & quality evaluation
│   ├── __init__.py
│   └── quality.py           # Laplacian sharpness, brightness, contrast
│
├── recognition/             # Facial embedding & verification (Step 4)
│   └── __init__.py
│
├── inference/               # Hardware runtime abstraction
│   ├── __init__.py
│   ├── device.py            # Device resolution (AUTO, CPU, GPU fallback)
│   └── runtime.py           # ONNX Runtime session manager
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
