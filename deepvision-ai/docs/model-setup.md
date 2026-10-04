# DeepVision AI — Model Setup & Inference Deployment Guide

This guide documents the remaining steps to deploy production ONNX model weights into DeepVision AI and details the hardware constraints and environment compatibility evaluated during **Step 2**.

---

## 1. Current State & Completed Foundations

All architectural layers for face detection inference are established and validated:

1. **AI Engine Modular Layout**: Located under `backend/app/ai/` (submodules: `common`, `detection`, `inference`, `analysis`, `tracking`, `recognition`).
2. **Abstract Detector Interface**: `BaseFaceDetector` defining the lifecycle: `load_model()`, `detect()`, `unload()`, `is_loaded`.
3. **Hardware Runtime Abstraction**: `InferenceRuntime` wrapping ONNX Runtime with hardware resolution (`DeviceType.AUTO`, `DeviceType.CPU`, `DeviceType.GPU`) and zero-crash fallback logic.
4. **Model Output Adapter**: `SCRFDAdapter` in `app.ai.detection.adapter` decoding multi-stride feature anchors (strides 8, 16, 32), bounding box regression offsets, 5 facial landmarks, and Non-Maximum Suppression (NMS).
5. **Preprocessing Pipeline**: `ImagePreprocessor` with aspect-ratio letterboxing, NCHW normalization, coordinate unpadding, and domain exception handling (`CorruptedImageException`, `ImageTooLargeException`, etc.).
6. **Application Service Layer**: `AIService` (`detect_faces`) ensuring the API layer never touches raw model weights or tensor sessions directly.

---

## 2. Environment Compatibility & Hardware Status

During hardware diagnostic inspection on this host:
- **Host CPU**: Intel Core i5-8365U (4 physical cores, 8 logical threads).
- **Available RAM**: ~16 GB Total (~1.0 GB currently free).
- **GPU Architecture**: Intel UHD Graphics 620 (Integrated graphics, shared memory).
- **CUDA / NVIDIA Discrete GPU**: **Unavailable** (`nvidia-smi` not found; discrete CUDA hardware is not present on this host).
- **Active Execution Provider**: `CPUExecutionProvider` configured with multi-threaded intra-op parallelism (4 worker threads allocated).

### Why Model Binaries Were Not Force-Installed:
1. **Source Control Integrity**: Committing multi-megabyte model binary files (`.onnx`, `.pt`) directly into Git bloats repository history and degrades clone performance.
2. **Network Decoupling**: Build environments and local developer setups should not depend on unauthenticated or brittle remote weight downloads during phase initialization.
3. **Architectural Verification First**: The complete software pipeline (preprocessing, schemas, telemetry, lifecycle) is fully verifiable via the integrated baseline adapter and mock tests before placing binary weights.

---

## 3. Step-by-Step Production Model Placement

Once ready to deploy live SCRFD weights:

### Step 1: Download Target Model Weights
Download the official SCRFD ONNX model (recommended: **SCRFD-500M** for lightweight CPU deployments, or **SCRFD-2.5G** for higher accuracy):

- **Target File**: `scrfd_500m_bnkps_shape640x640.onnx`
- **Official Source**: [InsightFace Model Zoo (SCRFD)](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- **Model Size**: ~1.7 MB (SCRFD-500M) / ~3.3 MB (SCRFD-2.5G)
- **Input Dimensions**: `1 x 3 x 640 x 640` (NCHW, Float32 in `[0.0, 1.0]`)
- **Integrated Outputs**: Multi-stride scores (8, 16, 32), bounding box offsets, and 5 facial landmark regression points.

### Step 2: Place Binary in Models Directory
Save the file to:
```
deepvision-ai/models/face_detection/scrfd_500m_bnkps_shape640x640.onnx
```
*(This path is automatically ignored by Git via `.gitignore`)*.

### Step 3: Configure Environment
Set the model path and target device in your `.env` file:
```ini
AI_DEVICE=auto
AI_MODEL_NAME=scrfd_500m
AI_MODEL_VERSION=1.0.0
AI_CONFIDENCE_THRESHOLD=0.50
AI_ENABLE_GPU=false
```

### Step 4: Verify Live Weights
Run the verification script:
```bash
python scripts/verify_ai_pipeline.py
```
`FaceDetector` will detect the physical `.onnx` file, compile an active `ort.InferenceSession`, and forward incoming frames through `InferenceRuntime` and `SCRFDAdapter`.
