# DeepVision AI

**Advanced AI-Powered Face Detection, Video Analysis & Real-Time Vision Platform**

---

## 📌 Project Description

**DeepVision AI** is an enterprise-ready, high-performance computer vision system designed for intelligent facial detection, multi-face tracking, media stream analytics, and authorized vision intelligence. The platform is structured around clean modular architecture, async processing pipelines, and cloud-ready infrastructure.

---

## 🎯 Vision

DeepVision AI is being developed as a **modular AI computer-vision platform**. 

Rather than a monolithic computer-vision script, the architecture separates core concerns:
- **Presentation Layer**: Responsive Next.js frontend with real-time streaming capabilities.
- **API & Orchestration**: Asynchronous FastAPI service handling validated requests, authentication, and dispatching.
- **Database & Persistence**: Enterprise PostgreSQL persistence layer powered by SQLAlchemy 2.0.
- **Cache & Message Broker**: Redis connection foundation for job scheduling, stream chunking, and caching.
- **AI Processing Engines**: Decoupled detection, tracking, analysis, and inference engines that can be upgraded or scaled independently.

---

## 🧠 AI Engine

DeepVision AI features a decoupled, production-grade computer vision pipeline. The architecture ensures that web controllers, model interfaces, hardware inference sessions, and neural network weights remain strictly isolated.

### Inference Pipeline

Every detection request executes through a deterministic 9-stage processing flow:

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

### Pipeline Stages Explained

1. **Image Upload**: The image payload is received via HTTP `POST /api/v1/ai/detect` (`multipart/form-data`) or in-memory buffer. Processed 100% in volatile memory without permanent disk persistence.
2. **Validation**: Enforces multi-layer defense-in-depth: MIME whitelist (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`), binary magic number verification, payload size cap (10 MB), maximum dimensions ($4096 \times 4096\text{px}$), and untrusted filename sanitization.
3. **Preprocessing**: The raw image is decoded into an RGB uint8 array and resized using symmetric aspect-ratio letterboxing to prevent distortion. Formats such as Grayscale and RGBA are cleanly converted to 3-channel RGB.
4. **AI Inference**: The active runtime (OpenCV FaceDetectorYN / YuNet C++ SIMD backend or ONNX Runtime) evaluates the model using hardware acceleration (CUDA, DirectML) with an automatic zero-crash fallback to CPU execution.
5. **Face Detection**: Multi-scale feature extraction across strides 8, 16, and 32 yields candidate face proposals, raw bounding boxes, and 5 facial landmark keypoints.
6. **Confidence Filtering**: Prunes candidate proposals whose confidence scores fall below the threshold (global `AI_CONFIDENCE_THRESHOLD` or per-request override `?confidence_threshold=0.75`).
7. **NMS / Postprocessing**: Applies Non-Maximum Suppression (IoU $\le 0.30$) to remove duplicate overlapping detections, preserving the highest-confidence candidate. Inverts letterbox scaling and padding back to native image pixel space.
8. **Bounding Box Validation**: Validates geometric bounds ($x_1 \ge 0, y_1 \ge 0, x_2 \le W, y_2 \le H, x_2 > x_1, y_2 > y_1$), clips partial overflows to image boundaries, and discards degenerate candidate boxes.
9. **Structured JSON Result**: Returns a strongly-typed Pydantic v2 response (`AIDetectionResponse`) containing absolute bounding boxes, confidence scores, landmarks, and granular stage latency telemetry. Zero fabricated detections are ever returned.

---

## 🚦 Current Status

### **Step 1 — Foundation** ✅ `Completed`
* Core development environment, project hierarchy, database/cache layers, containerization, and foundational health endpoints verified.

### **Step 2 — AI Engine & Inference Foundation** ✅ `Completed`
* **Model Selection**: SCRFD & YuNet selected for multi-scale small-face recall, ONNX compatibility, and permissive licensing ([Documentation](docs/ai-model-selection.md)).
* **Modular AI Architecture**: Decoupled packages under `app/ai/` (`detection`, `tracking`, `analysis`, `inference`, `common`).
* **Hardware Abstraction**: Dynamic `AUTO`, `CPU`, `GPU` device resolution with zero-crash CPU fallback.
* **Detector Contract**: Typed `BaseFaceDetector` lifecycle (`load_model`, `detect`, `unload`).
* **Result Schemas**: Strongly typed Pydantic v2 schemas for bounding boxes, landmarks, quality scores, and responses.
* **Image Preprocessing**: Detector-agnostic preprocessing with aspect-ratio letterboxing and coordinate inversion.
* **Model Management**: Structure, version tracking, integrity verification, and Git exclusion policies ([Documentation](docs/model-management.md)).
* **Service Layer**: `AIService` orchestrator decoupling API controllers from model binaries.
* **Development Endpoint**: `POST /api/v1/ai/detect` with input validation, zero-fabrication enforcement, and `GET /api/v1/ai/status`.
* **Error Handling & Sanitization**: Clean AI exception hierarchy (`ModelLoadError`, `InferenceError`, `InvalidImageError`, etc.) mapped to HTTP status codes with zero stack trace exposure.
* **Structured Telemetry**: JSON-formatted inference performance logging with `request_id` tracing and zero raw payload logging.

### **Step 3 — Real Face Detection Implementation** ✅ `Completed`
* **Operational AI Engine**: Loaded production YuNet ONNX model (`face_detection_yunet_2023mar.onnx`, 232 KB) via native SIMD C++ backend and ONNX Runtime.
* **Real Bounding Boxes & Confidence**: Returns precise pixel coordinates, face crops, and genuine confidence scores directly from neural network activations.
* **5 Facial Landmarks**: Accurate localization of right eye, left eye, nose tip, mouth right corner, and mouth left corner.
* **Face Quality Analytics**: Computes Laplacian variance sharpness, perceptual brightness, contrast, and crop dimensions for every detected face.
* **Zero Fabrication Policy**: Mathematically guarantees zero bounding boxes fabricated for empty, blank, or random Gaussian noise images.
* **Automatic Model Discovery**: Dynamic resolution across `models/face_detection/`, environment variables, and relative development paths.
* **Live API Verification**: Validated on real portrait image (`sample_face.jpg`) with ~17 ms CPU latency.
* **Test Suite**: 146/146 automated unit and integration tests passing.

### **Step 4 — Face Landmarks & Quality Analysis** ✅ `Completed`
* **Landmark Strategy**: Formally documented 5-point native YuNet landmark strategy in `docs/face-landmark-model-selection.md`.
* **Modular Analysis Layer**: Implemented decoupled modules under `app/ai/analysis/`:
  - `landmarks.py`: 5-point landmark geometry (inter-pupillary distance, eye tilt angle, bilateral symmetry ratio).
  - `pose.py`: 3D Head pose estimation (Yaw, Pitch, Roll Euler angles, frontality score, and pose categories via `cv2.solvePnP` with 3D anthropometric model).
  - `metrics.py`: Blur/clarity Laplacian variance, illumination/exposure diagnostics, and face resolution/scale metrics.
  - `quality.py`: Weighted composite quality scoring ($0.0 \dots 1.0$), qualitative grading (`EXCELLENT` ... `UNUSABLE`), and usability validation.
  - `schemas.py`: Strongly typed Pydantic v2 schemas for all diagnostic metrics.
* **REST API Endpoint**: Added `POST /api/v1/ai/analyze` returning complete `AIAnalysisResponse` with granular per-face diagnostics.
* **Backward Compatibility**: Exact preservation of legacy `FaceQualityScore` and Step 2/3 contracts.
* **Automated Test Suite**: 176/176 tests passing (30 new dedicated Step 4 tests).

### Current Capabilities

✓ Real face detection
✓ Multiple-face detection
✓ Bounding boxes
✓ Confidence scores
✓ Facial landmarks
✓ Blur analysis
✓ Brightness analysis
✓ Contrast analysis
✓ Face-size analysis
✓ Basic pose estimation
✓ Face quality score

#### Detailed Capabilities Breakdown
* **✓ Real Face Detection**: Production YuNet & SCRFD ONNX models operating on genuine photos with real activations.
* **✓ Multiple-Face Detection**: Detects and localizes multiple faces across varying scales and crowd densities in a single forward pass.
* **✓ Bounding Boxes**: Absolute pixel coordinates validated and bounds-clipped to image dimensions.
* **✓ Confidence Scores**: Authentic probability scores ($0.0 \dots 1.0$) calculated directly from neural activations.
* **✓ Facial Landmarks**: 5 canonical landmarks (eyes, nose, mouth corners) with sub-pixel localization, IPD, and symmetry metrics.
* **✓ Blur Analysis**: Laplacian variance spatial frequency analysis and continuous blur scoring (`sharp`, `slightly_blurred`, `blurry`).
* **✓ Brightness Analysis**: ITU-R BT.601 photopic luminance evaluation and exposure status (`too_dark`, `acceptable`, `too_bright`).
* **✓ Contrast Analysis**: Standard deviation luminance dynamic range diagnostics (`low_contrast`, `acceptable`, `high_contrast`).
* **✓ Face-Size Analysis**: Bounding box width, height, aspect ratio, area coverage ratio, and biometric sufficiency check.
* **✓ Basic Pose Estimation**: Perspective-n-Point Euler rotation angles (Yaw, Pitch, Roll) and frontality score (degrades gracefully to `null` on degenerate geometry).
* **✓ Face Quality Score**: Explainable, transparent 0–100 composite score combining sharpness, brightness, contrast, size, and landmarks with bottleneck gating.
* **✓ Granular Latency Profiling**: Real measured latencies for detection, landmarks, quality, pose, and total analysis.
* **✓ Typed Configuration & Privacy**: Environment settings validation, zero-secret exposure, and zero permanent biometric disk storage.

#### ⚠️ What is NOT Implemented Yet
* ❌ **Identity Recognition / Biometric Matching**: No facial embeddings extraction, identity database matching, or biometric identification (`face_id: null`).
* ❌ **Multi-Frame Video Tracking**: No temporal trajectory tracking across video frames (e.g. ByteTrack, SORT) or persistent track IDs.
* ❌ **Live Webcam Streaming WebSocket**: Real-time live webcam frame-by-frame streaming over WebSocket is not yet connected to the deep face detection model.
* ❌ **Liveness & Anti-Spoofing Detection**: No 2D/3D print attack, screen replay, or silicon mask anti-spoofing verification models.
* ❌ **Final Production Dashboard**: The analytics UI dashboard is not built yet.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | **Next.js** + **TypeScript** + **Tailwind CSS** |
| **Backend** | **FastAPI** + **Python 3.12** + **Uvicorn** |
| **AI / Computer Vision** | **ONNX Runtime** + **NumPy** + **OpenCV** + **Pillow** |
| **Database** | **PostgreSQL** (with SQLAlchemy 2.0 ORM & psycopg 3) |
| **Cache / Queue** | **Redis** |
| **Infrastructure** | **Docker** & **Docker Compose** |
| **Testing** | **pytest** + **pytest-anyio** + **httpx** |

---

## 📚 Technical Documentation

- 🏛️ [AI Engine Architecture](docs/ai-architecture.md) — Architectural design, 9-stage pipeline, Face Analysis layer, component breakdown, and error shielding.
- 🔬 [Face Analysis Methodology](docs/face-analysis.md) — Mathematical algorithms, landmark models, optical diagnostics, 3D pose, and 0–100 quality scoring.
- 🛡️ [Security & Privacy Architecture](docs/security-and-privacy.md) — 9 ingestion defense layers, non-identification guarantees, and zero-persistence controls.
- 🔬 [Face Landmark Model Selection](docs/face-landmark-model-selection.md) — Architectural evaluation and trade-off analysis of landmark extraction strategies.
- 🔬 [AI Model Selection Report](docs/ai-model-selection.md) — Evaluation of SCRFD, YuNet, RetinaFace, and YOLO-Face on WIDER FACE benchmarks.
- 📦 [Model Management Guide](docs/model-management.md) — Storage structure, versioning, download procedures, SHA-256 verification, and hardware variants.
- ⚙️ [Model Setup & Deployment](docs/model-setup.md) — Host hardware constraints, ONNX weights placement, and production instructions.

---

## 🚀 Quickstart (Local Development)

### 1. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 2. Run the Backend

```bash
cd backend
# Activate virtual environment
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

# Run development server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
* **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health Endpoint**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
* **AI Status Endpoint**: [http://127.0.0.1:8000/api/v1/ai/status](http://127.0.0.1:8000/api/v1/ai/status)

### 3. Run Backend Test Suite

```bash
cd backend
pytest tests -v
```

### 4. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```
* **Frontend Application**: [http://localhost:3000](http://localhost:3000)

### 5. Run via Docker Compose

```bash
docker compose up --build
```
