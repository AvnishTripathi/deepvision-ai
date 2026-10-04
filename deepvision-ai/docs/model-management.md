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

Model configurations and versions are tracked via typed settings and manifest metadata:

```bash
# Environment Configuration
AI_MODEL_NAME=scrfd_500m
AI_MODEL_VERSION=1.0.0
AI_DEVICE=auto
```

Each model directory supports an optional `manifest.json` describing artifact specifications:

```json
{
  "model_name": "scrfd_500m",
  "version": "1.0.0",
  "architecture": "SCRFD",
  "input_shape": [1, 3, 640, 640],
  "format": "ONNX",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "source_url": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx",
  "recommended_strides": [8, 16, 32],
  "date_added": "2026-10-04"
}
```

---

## 4. Model Download Procedure

Model binaries can be acquired through the platform's automated setup script or via direct download:

### Automated Download (Future CLI Utility)
```bash
python scripts/download_model.py --model scrfd_500m --destination models/face_detection/
```

### Manual Download
Download the official SCRFD ONNX weights and place them into the designated directory:
```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx" `
  -OutFile "models/face_detection/scrfd_500m.onnx"

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

## 6. CPU and GPU Variant Handling

DeepVision AI utilizes standard ONNX model graphs that execute seamlessly across both CPU and GPU execution providers without requiring separate model binaries:

* **Universal Graph Execution**: A single `.onnx` graph runs on CPU (`CPUExecutionProvider`), NVIDIA GPU (`CUDAExecutionProvider`, `TensorrtExecutionProvider`), or DirectML on Windows (`DmlExecutionProvider`).
* **Hardware Provider Fallback**: The inference runtime dynamically inspects available hardware. If a GPU driver is missing, it falls back to multi-threaded CPU execution with zero downtime or crashes.
* **Quantization & Edge Optimization**: In low-resource environments (e.g. edge or pure CPU servers), an INT8 quantized model variant (`scrfd_500m_int8.onnx`) can be deployed to reduce memory usage by ~50% and accelerate integer arithmetic.
