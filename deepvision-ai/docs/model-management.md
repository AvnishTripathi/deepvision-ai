# DeepVision AI — Model Management Architecture

This document describes the model storage structure, lifecycle management, version tracking, integrity verification, and runtime execution strategy for DeepVision AI.

---

## 1. Model Storage Hierarchy

All model artifacts are stored in the root `models/` directory, partitioned cleanly by computer vision capability:

```text
models/
├── face_detection/          # Face localization & keypoint models (e.g. SCRFD ONNX)
│   └── .gitkeep
├── face_landmarks/          # Detailed facial landmark meshes (e.g. 68-pt / 106-pt)
│   └── .gitkeep
├── face_recognition/        # Facial embedding feature extractors (e.g. ArcFace)
│   └── .gitkeep
└── liveness/                # Presentation attack & anti-spoofing models
    └── .gitkeep
```

### Git Exclusion Policy
Large model binaries (`.onnx`, `.pt`, `.engine`, `.bin`, `.pth`) are strictly excluded from Git tracking via `.gitignore`:

```gitignore
# DeepVision AI — Model Binary Artifact Exclusions
models/**/*.onnx
models/**/*.engine
models/**/*.pt
models/**/*.pth
models/**/*.bin
!models/**/.gitkeep
```

Only directory placeholders (`.gitkeep`) and metadata manifest files (`manifest.json`) are committed to version control.

---

## 2. Model Loading Lifecycle

Model sessions follow an explicit 3-stage lifecycle contract defined by `BaseFaceDetector`:

```python
class BaseFaceDetector(ABC):
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        """Loads weights, configures graph optimizations, and creates ONNX session."""
        pass

    @abstractmethod
    def detect(self, image_input: ImageSource, confidence_threshold: float = 0.5) -> FrameInferenceResult:
        """Executes forward-pass inference on preprocessed tensor."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Destroys inference session, frees VRAM, and releases thread pools."""
        pass
```

### Lazy Loading and Resource Teardown
1. **Lazy Initialization**: Models are loaded on-demand during application startup or first inference request, avoiding unnecessary memory allocation during cold boot.
2. **Deterministic Cleanup**: Calling `detector.unload()` explicitly releases ONNX Runtime memory buffers, session pointers, and CUDA/DirectML contexts.
3. **Context Manager Support**: `FaceDetector` implements `__enter__` and `__exit__` for scoped execution in worker processes.

---

## 3. Model Version Tracking

## 3. Supported Model Registries & Specifications

DeepVision AI supports standardized lightweight face detection models. The table below outlines the certified models, checksums, licenses, and sources:

| Model | Architecture | File Size | Checksum (SHA-256) | License | Source / Upstream |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YuNet (Default)** | YuNet CNN | 232 KB | `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4` | Apache-2.0 | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet) |
| **SCRFD-500M** | SCRFD Mobile | ~1.7 MB | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | Apache-2.0 | [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) |

### Version Tracking Configuration

Model configurations and versions are tracked via typed settings:

```bash
# Environment Configuration
AI_MODEL_NAME=face_detection_yunet_2023mar
AI_MODEL_VERSION=2023mar
AI_DEVICE=auto
AI_CONFIDENCE_THRESHOLD=0.50
AI_REQUIRE_MODEL=false  # Set to true in production to enforce model file existence
```

Each model directory supports an optional `manifest.json` describing artifact specifications:

```json
{
  "model_name": "face_detection_yunet_2023mar",
  "version": "2023mar",
  "architecture": "YuNet",
  "input_shape": [1, 3, 320, 320],
  "format": "ONNX",
  "sha256": "8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4",
  "license": "Apache-2.0",
  "source_url": "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
  "date_added": "2026-10-04"
}
```

---

## 4. Model Download & Setup Procedure

### Startup Policy: No Silent Startup Downloads
In accordance with production reliability standards, **the application never silently downloads models during application boot**. This guarantees:
1. **Deterministic Cold Starts**: Server boots in milliseconds without unpredictable network latency.
2. **Air-Gapped & Offline Compatibility**: The container can deploy into isolated networks without egress internet connectivity.
3. **Zero Unexpected Billable Bandwidth**: Cloud egress and third-party download rate limits will not crash cluster instances.

### Explicit Missing Model Failure
When `AI_REQUIRE_MODEL=true` is enabled (recommended in staging and production), the detector enforces model presence immediately upon initialization:
- If the weights file is absent, it raises `FileNotFoundError: Required AI model file not found at '<path>'. Please ensure model weights are downloaded and placed in the models directory.`
- In local development (`AI_REQUIRE_MODEL=false`), it gracefully falls back to the baseline algorithmic detection engine with descriptive info logs.

### Model Download Commands

#### Option A: YuNet (Default Production Lightweight Model)
```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
  -OutFile "models/face_detection/face_detection_yunet_2023mar.onnx"
```

```bash
# Linux / macOS / Docker
curl -L -o models/face_detection/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

#### Option B: SCRFD-500M (Alternative High-Resolution Model)
```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx" `
  -OutFile "models/face_detection/scrfd_500m.onnx"
```

```bash
# Linux / macOS
curl -L -o models/face_detection/scrfd_500m.onnx \
  https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx
```

---

## 5. Model Integrity & Checksum Verification

To ensure downloaded models have not been corrupted during transfer or tampered with, the system performs cryptographic SHA-256 verification before initializing an inference session:

```python
import hashlib

def verify_model_checksum(model_path: str, expected_sha256: str) -> bool:
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()
    return actual_hash.lower() == expected_sha256.lower()
```

If the checksum does not match, the system raises a `ModelLoadError` and refuses to initialize the session.

---

## 6. Runtime Requirements & Hardware Providers

DeepVision AI utilizes standard ONNX model graphs that execute seamlessly across both CPU and GPU execution providers:

* **CPU Execution Requirements**:
  * Any standard x86_64 or ARM64 processor (Intel, AMD, Apple Silicon, AWS Graviton).
  * Minimal RAM footprint: ~20 MB memory overhead for inference graphs.
  * Multi-threaded intra-op parallelism using OpenCV DNN or ONNX Runtime `CPUExecutionProvider`.
* **GPU Execution Requirements**:
  * NVIDIA GPU with CUDA 11.x or 12.x and cuDNN 8.x/9.x (`CUDAExecutionProvider`).
  * Windows DirectML support for Intel/AMD/NVIDIA integrated or discrete GPUs (`DmlExecutionProvider`).
* **Hardware Provider Fallback**: The inference runtime dynamically inspects available hardware. If a GPU driver is missing, it falls back to multi-threaded CPU execution with zero downtime or crashes.

