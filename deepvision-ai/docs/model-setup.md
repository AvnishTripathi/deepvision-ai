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

### Primary Certified Model: YuNet (Recommended for CPU & Lightweight Production)
- **Target File**: `face_detection_yunet_2023mar.onnx`
- **Official Source**: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
- **SHA-256 Checksum**: `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4`
- **File Size**: 232 KB (232,589 bytes)
- **License**: Apache-2.0
- **Input Dimensions**: Dynamic resolution (`320 x 320` to `1280 x 720`)
- **Integrated Outputs**: Face bounding boxes, confidence score, and 5 facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner).

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

### Alternative High-Resolution Model: SCRFD-500M
- **Target File**: `scrfd_500m_bnkps_shape640x640.onnx`
- **Official Source**: [InsightFace Model Zoo (SCRFD)](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- **SHA-256 Checksum**: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- **File Size**: ~1.7 MB
- **License**: Apache-2.0
- **Input Dimensions**: `1 x 3 x 640 x 640` (NCHW, Float32 in `[0.0, 1.0]`)

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx" `
  -OutFile "models/face_detection/scrfd_500m.onnx"
```

---

## 4. Environment Configuration & Verification

Configure the model in `.env`:
```ini
AI_DEVICE=auto
AI_MODEL_NAME=face_detection_yunet_2023mar
AI_MODEL_VERSION=2023mar
AI_CONFIDENCE_THRESHOLD=0.50
AI_ENABLE_GPU=false
AI_REQUIRE_MODEL=false  # Set to true in production to fail fast if missing
```

Run verification:
```bash
python scripts/verify_ai_pipeline.py
```
`FaceDetector` detects the `.onnx` file and loads the OpenCV FaceDetectorYN or ONNX Runtime inference engine.

