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
├── backend/                 # FastAPI Python server
│   ├── main.py              # App entry point
│   ├── config.py            # Settings (env vars)
│   ├── database.py          # SQLAlchemy setup
│   ├── models.py            # DB models
│   ├── schemas.py           # Pydantic schemas
│   ├── auth.py              # JWT authentication
│   ├── routers/
│   │   ├── auth_router.py   # /api/v1/auth endpoints
│   │   ├── detect_router.py # /api/v1/detect endpoints + WebSocket
│   │   └── history_router.py# /api/v1/history endpoints
│   └── requirements.txt
│
├── ml_engine/               # AI/ML deepfake detection
│   ├── detector.py          # Main orchestrator
│   ├── face_detector.py     # MTCNN face detection
│   ├── deepfake_classifier.py # Multi-signal classifier
│   ├── heatmap_generator.py # Visualization
│   └── model_manager.py     # Model loading & caching
│
├── frontend/                # React 18 + TypeScript
│   ├── src/
│   │   ├── pages/           # All pages
│   │   ├── components/      # Reusable components
│   │   ├── api/client.ts    # API + WebSocket client
│   │   └── store/           # Zustand state
│   └── package.json
│
├── docker-compose.yml       # Full stack Docker
├── setup.bat                # Windows setup script
└── .env                     # Configuration
```

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
