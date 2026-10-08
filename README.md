# 🛡️ DeepFaceGuard

> **Industry-grade Deepfake Detection Platform** — Detect deepfakes in photos, videos & live webcam with AI

![DeepFaceGuard](https://img.shields.io/badge/AI%20Powered-EfficientNet--B7-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![React](https://img.shields.io/badge/Frontend-React%2018-61dafb)
![Python](https://img.shields.io/badge/Python-3.11+-yellow)

---

## ✨ Features

- 📸 **Photo Detection** — Upload any image, get deepfake analysis in < 500ms
- 🎥 **Video Analysis** — Frame-by-frame analysis with timeline heatmap
- 🔴 **Live Webcam** — Real-time deepfake detection via WebSocket stream
- 🧠 **AI Explainability** — GradCAM heatmaps showing suspicious regions
- 📊 **Signal Analysis** — Frequency, blur, color & compression artifact scores
- 🔐 **Auth System** — JWT-based registration & login
- 📜 **Detection History** — Full history of past detections
- 🌐 **REST API** — Full API access for programmatic use

---

## 🚀 Quick Start (2 steps)

### Option A: Manual (Recommended for development)

**Step 1 — Backend**
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

**Step 2 — Frontend** (new terminal)
```bash
cd frontend
npm install
npm run dev
```

Open → **http://localhost:5173**

API docs → **http://localhost:8000/docs**

---

### Option B: One-click setup
```bash
# Double-click setup.bat
```

---

## 📁 Project Structure

```
DeepFaceGuard/
├── deepvision-ai/           # Production DeepVision AI microservice
│   ├── backend/             # FastAPI computer vision server
│   ├── models/              # Neural network weights (YuNet, ONNX)
│   ├── docs/                # Architecture & model selection docs
│   └── scripts/             # Development visualization utilities
│
├── backend/                 # Platform API & database server
├── frontend/                # React 18 + TypeScript + Vite UI
├── ml_engine/               # Multi-signal deepfake classification
└── docker-compose.yml       # Full stack Docker
```

---

## 🧠 DeepVision AI Face Detection Pipeline

Every detection request executes through a deterministic 9-stage pipeline:

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

### 🌟 Current Capabilities (End of Step 3)
* **Real Image Face Detection**: Real, production neural network inference using YuNet ONNX weights (`face_detection_yunet_2023mar.onnx`).
* **Multiple Faces**: Accurately localizes and detects multiple human faces across varying scales, orientations, and crowded scenes in a single forward pass.
* **Confidence Scores**: Authentic floating-point probability scores ($0.0 \dots 1.0$) computed directly from model activations.
* **Bounding Boxes**: Absolute, pixel-accurate bounding box coordinates ($x_1, y_1, x_2, y_2$) mapped directly to native image dimensions.
* **Configurable Threshold**: Globally configurable detection sensitivity via `AI_CONFIDENCE_THRESHOLD` (default `0.6`) and per-request dynamic overrides via query parameter `?confidence_threshold=`.
* **CPU/GPU Runtime Where Supported**: Resilient hardware abstraction executing reliably on CPU with SIMD optimizations, with automatic GPU enablement (CUDA, DirectML) where drivers and runtime execution providers exist.
* **Inference Timing**: Real measured latency profiling returned in API responses and logged, including granular breakdown (`preprocessing_time_ms`, `inference_time_ms`, `postprocessing_time_ms`, `total_processing_time_ms`).
* **API Access**: Hardened REST API endpoint `POST /api/v1/ai/detect` documented via OpenAPI / Swagger UI (`/docs`), input validation, and sanitized error schemas (`400`, `413`, `500`, `503`).
* **Development Visualization**: Offline developer utility (`scripts/visualize_detections.py` and `app/ai/visualization/annotator.py`) to annotate images with bounding boxes, confidence score badges, and 5-point facial landmarks without modifying the original input.

### ⚠️ What is NOT Implemented Yet
* ❌ **Identity Recognition / Biometric Matching**: No facial embeddings extraction, identity database matching, or biometric identification is active (`face_id: null`).
* ❌ **Multi-Frame Video Tracking**: No temporal trajectory tracking across video frames (e.g. ByteTrack, SORT) or track ID continuity.
* ❌ **Live Webcam Streaming WebSocket**: Real-time live webcam frame-by-frame streaming over WebSocket is not yet connected to the deep face detection model.
* ❌ **Liveness & Anti-Spoofing Detection**: No 2D/3D print attack, screen replay, or silicon mask anti-spoofing verification models.
* ❌ **Facial Attribute Analysis**: No estimation of age, gender, emotion, head pose angles (yaw, pitch, roll), or facial landmarks beyond the core 5 keypoints.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login` | Get JWT token |
| GET | `/api/v1/auth/me` | Get current user |
| POST | `/api/v1/detect/image` | Detect deepfake in image |
| POST | `/api/v1/detect/video` | Queue video analysis |
| GET | `/api/v1/detect/status/{id}` | Poll job status |
| GET | `/api/v1/detect/result/{id}` | Get full result |
| WS | `/api/v1/detect/ws/stream` | Real-time webcam stream |
| GET | `/api/v1/history` | Get detection history |
| GET | `/api/v1/stats` | Usage statistics |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## 🧠 ML Detection Signals

| Signal | Weight | What it detects |
|--------|--------|-----------------|
| Frequency Analysis | 35% | GAN checkerboard artifacts in FFT |
| Blur Analysis | 20% | Unnaturally smooth skin (Laplacian) |
| Color Consistency | 20% | Paste boundary color mismatch |
| Compression Artifacts | 10% | Re-compression block artifacts |
| Neural Features | 15% | EfficientNet-B7 anomaly detection |

**Labels:** `REAL` (< 35%) | `SUSPICIOUS` (35-65%) | `DEEPFAKE` (> 65%)

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | SQLite | Database connection string |
| `SECRET_KEY` | (required) | JWT signing key |
| `UPLOAD_DIR` | `./uploads` | File upload directory |
| `MAX_IMAGE_SIZE_MB` | 50 | Max image upload size |
| `MAX_VIDEO_SIZE_MB` | 500 | Max video upload size |

---

*Built with ❤️ for security professionals, researchers & journalists*
