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

Every detection request executes through a deterministic 7-stage processing flow:

```text
Image
  ↓
Validation
  ↓
Preprocessing
  ↓
Inference Runtime
  ↓
Face Detector
  ↓
Structured Detection Result
  ↓
API Response
```

### Pipeline Stages Explained

1. **Image**: An image payload is received via HTTP upload (`multipart/form-data`) or in-memory byte buffer. No image files are saved permanently to disk.
2. **Validation**: The image undergoes strict format validation (JPEG, PNG, WebP, BMP), size checks (up to 10 MB limit), dimension bounds ($\le 4096\text{px}$), and corruption verification. Invalid or corrupt inputs are rejected cleanly with standard HTTP client errors (`400 Bad Request`, `413 Content Too Large`).
3. **Preprocessing**: The raw image is decoded into an RGB uint8 array and resized using symmetric aspect-ratio letterboxing to prevent distortion. It is then normalized and converted into an optimized NCHW float32 tensor $(1, 3, 640, 640)$.
4. **Inference Runtime**: An ONNX Runtime session executes the model using hardware acceleration (CUDA, DirectML) with an automatic, zero-crash fallback to multi-threaded CPU execution.
5. **Face Detector**: The detector parses multi-scale feature pyramids across strides 8, 16, and 32 to detect faces of varying scales. It applies Non-Maximum Suppression (NMS) to eliminate duplicate candidate proposals, extracts 5 facial landmarks, and inverts letterbox coordinates back into original image space.
6. **Structured Detection Result**: Detections are mapped to strongly-typed Pydantic v2 schemas (`BoundingBox`, `FaceDetection`, `FrameInferenceResult`) complete with latency profiling and face quality diagnostics. No fabricated detections are ever returned.
7. **API Response**: The response is returned as clean JSON alongside trace headers (`X-Request-ID`). Structured performance telemetry is logged without exposing sensitive image contents or internal stack traces.

---

## 🚦 Current Status

### **Step 1 — Foundation** ✅ `Completed`
* Core development environment, project hierarchy, database/cache layers, containerization, and foundational health endpoints verified.

### **Step 2 — AI Engine & Inference Foundation** ✅ `Completed`
* **Model Selection**: SCRFD selected for multi-scale small-face recall, ONNX compatibility, and permissive licensing ([Documentation](docs/ai-model-selection.md)).
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
* **Test Suite**: 84/84 unit and integration tests passing.

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

- 🏛️ [AI Engine Architecture](docs/ai-architecture.md) — Architectural design, 7-stage pipeline, component breakdown, and error shielding.
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
