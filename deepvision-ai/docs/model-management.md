# DeepVision AI — Model Management Architecture

This document defines the storage layout, lifecycle management, versioning, automated retrieval, integrity verification, and runtime execution strategies for machine learning models across the DeepVision AI platform.

---

## 1. Directory Structure & Storage Strategy

Model weights are categorized by computer vision task in dedicated directories under `models/`:

```
models/
├── face_detection/
│   ├── .gitkeep
│   └── scrfd_500m_bnkps_shape640x640.onnx  (Git-ignored)
├── face_landmarks/
│   ├── .gitkeep
│   └── landmark_2d_106_shape192x192.onnx   (Git-ignored)
├── face_recognition/
│   ├── .gitkeep
│   └── arcface_mobilefacenet_v1.onnx       (Git-ignored)
└── liveness/
    ├── .gitkeep
    └── minifasnet_v2_shape80x80.onnx       (Git-ignored)
```

### Git Policy
- **Weights Exclusion**: Large model binaries (`*.onnx`, `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.engine`) are strictly excluded from source control in `.gitignore`.
- **Directory Preservation**: Subdirectory structures are tracked in Git using lightweight `.gitkeep` anchor files.
- **Local Cache & Docker Mounts**: In containerized environments, the host `models/` directory is mounted read-only into `/app/models` to avoid inflating image layers.

---

## 2. Model Loading Lifecycle

Model instantiation and execution are decoupled from API handlers and isolated within the AI engine:

```
[API Endpoint / Service]
         │
         ▼
[FaceDetector Interface] (Lifecycle controller: load_model, detect, unload)
         │
         ▼
[InferenceRuntime]       (ONNX Runtime SessionOptions & Execution Providers)
         │
         ▼
[Hardware Device]        (CPU multi-threading / GPU execution provider)
```

### Key Principles:
1. **Lazy vs. Eager Loading**:
   - In production worker processes, models can be loaded eagerly during application startup (`lifespan` hook) to absorb graph compilation latency before accepting client requests.
   - In development and test environments, models are loaded lazily upon first invocation.
2. **Session Reuse**:
   - `InferenceRuntime` holds an active `ort.InferenceSession` singleton per model, reusing thread pools across sequential frames without recompilation.
3. **Explicit Memory Deallocation**:
   - `unload()` clears the session reference, releases CPU thread pools, and invokes garbage collection to free host RAM and GPU VRAM.
   - The detector supports standard Python context managers (`with FaceDetector(...) as detector:`) for deterministic cleanup in batch pipelines.

---

## 3. Model Versioning & Manifest Strategy

DeepVision AI utilizes an explicit manifest file (`models/manifest.json`) to track model definitions, operational thresholds, and cryptographic hashes.

### Naming Convention
Model binaries adhere to a standard semantic identifier:
```
<task>_<architecture>_<variant>_<opset>_v<semver>.onnx
```
*Example*: `face_detection_scrfd_500m_bnkps_shape640x640_v1.0.0.onnx`

### Manifest Schema Example (`manifest.json`):
```json
{
  "version": "1.0.0",
  "models": {
    "face_detection": {
      "default": "scrfd_500m",
      "available": {
        "scrfd_500m": {
          "version": "1.0.0",
          "filename": "scrfd_500m_bnkps_shape640x640.onnx",
          "sha256": "3fa98e826b647f1fbc42d5f3083e20e6f6cb1928ab16c84c3c3deba0bfa813d1",
          "size_bytes": 1723450,
          "input_shape": [1, 3, 640, 640],
          "mean": [0.0, 0.0, 0.0],
          "std": [1.0, 1.0, 1.0],
          "confidence_threshold": 0.5,
          "nms_threshold": 0.4,
          "source_url": "https://deepvision-assets.s3.amazonaws.com/models/v1/scrfd_500m_bnkps_shape640x640.onnx"
        }
      }
    }
  }
}
```

---

## 4. Model Download & Distribution Strategy

To avoid requiring massive binary check-ins while ensuring reproducible deployments, models can be downloaded on-demand or as an automated build step.

### Automated Download Pipeline:
1. **Target Evaluation**: The download runner parses `manifest.json` to identify active model requirements.
2. **Local Cache Check**: If the target `.onnx` exists locally and matches the SHA-256 hash, download is skipped.
3. **Atomic Streaming Download**:
   - The file is streamed from a secure artifact registry (e.g., S3, Google Cloud Storage, or Hugging Face release assets) into a temporary staging file (`<model_name>.tmp`).
   - Upon successful stream completion and integrity validation, the file is renamed atomically to its final `.onnx` destination.
   - Prevents corrupted or partial models from entering the runtime.

---

## 5. Model Integrity & Checksum Verification

To protect against corrupted downloads, incomplete file transfers, or unauthorized file tampering, every model file is cryptographically validated using SHA-256:

```python
import hashlib

def verify_model_integrity(file_path: str, expected_sha256: str) -> bool:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    
    calculated = sha256_hash.hexdigest().lower()
    return calculated == expected_sha256.lower()
```

If the calculated hash mismatches the manifest:
- The corrupted file is quarantined or deleted.
- An alert is logged, and the application halts model initialization rather than executing an untrusted binary.

---

## 6. CPU vs. GPU Variants & Optimization

DeepVision AI utilizes ONNX (Open Neural Network Exchange) to provide unified model weight portability across computing platforms without maintaining separate model codebases.

### Execution Strategies:

| Hardware Environment | Target Precision | ONNX Runtime Provider | Optimization Strategy |
| :--- | :--- | :--- | :--- |
| **Standard CPU** (Intel/AMD) | FP32 / INT8 | `CPUExecutionProvider` | Multi-threading (`intra_op_num_threads` tuned to 2–4 physical cores), graph optimizations (`ORT_ENABLE_ALL`). |
| **Windows Desktop/Laptop** | FP32 | `DmlExecutionProvider` (DirectML) | Utilizes integrated or discrete GPUs (Intel UHD/Iris, AMD Radeon, NVIDIA GeForce) on Windows 10/11. |
| **NVIDIA Server/Workstation** | FP32 / FP16 | `CUDAExecutionProvider` / `TensorrtExecutionProvider` | Memory pre-allocation, CUDA stream concurrency, TensorRT engine compilation. |
| **Intel NUC / Edge VPU** | INT8 / FP16 | `OpenVINOExecutionProvider` | Vector Neural Network Instructions (VNNI), Intel iGPU / VPU hardware offload. |

### Dynamic Device Fallback:
As established in [`device.py`](file:///c:/Users/abhis/Music/Deep%20Face%20Detection/deepvision-ai/backend/app/ai/inference/device.py):
- `AI_DEVICE=auto`: Discovers available acceleration providers. If a GPU provider is missing or fails initialization, safely executes on `CPUExecutionProvider`.
- `AI_DEVICE=gpu`: Attempts GPU execution; if unavailable, issues an audit warning and transparently falls back to CPU without process termination.
