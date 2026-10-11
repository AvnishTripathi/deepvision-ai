# DeepVision AI — Frontend Architecture & AI Dashboard

This document details the architecture, design principles, component hierarchy, API integrations, and developer guidelines for the **DeepVision AI** frontend interface.

---

## 1. Product Overview & Identity

- **Product Name**: DeepVision AI
- **Tagline**: *See. Detect. Track. Analyze. Understand.*
- **Core Mission**: Provide a production-grade, intuitive, and accessible interface for computer vision engineers and operators to run real-time face detection, multi-face video tracking, 3D pose estimation, and optical diagnostics.
- **Backend Coupling**: Directly interfaces with the real FastAPI service powered by OpenCV DNN / YuNet ONNX and ByteTrack MOT. Zero mock detections or fabricated progress numbers are used.

---

## 2. Technology Stack

- **Framework**: Next.js 14 (App Router architecture with React 18)
- **Language**: TypeScript 5.6 (strict typing, zero `any` shortcuts)
- **Styling**: Tailwind CSS v3.4 + PostCSS with custom neural-vision color palette (`dark-950`, `dark-900`, `dark-850`, `brand-cyan`, `brand-blue`, `brand-violet`)
- **Icons**: Lucide React
- **Client Networking**: Native `fetch` with configurable timeout and error parsing (`src/lib/api/`)
- **Theme**: Persistent Dark AI Workspace / High-Contrast Light theme (`ThemeProvider` + localStorage)

---

## 3. Application Routes & Studios

### `GET /` — Landing Page & Platform Overview
- **Hero Banner**: Product tagline, value proposition, and primary call-to-action button (*Start Analyzing* navigating to `/image`).
- **6 Core Feature Pillars**:
  1. **Image Detection**: YuNet ONNX multi-scale face detection with sub-millisecond forward pass.
  2. **Video Tracking**: Multi-face tracking via ByteTrack and 2D Kalman motion forecasting.
  3. **Face Analysis**: 5-point facial landmark geometry and 3D head pose estimation (Yaw/Pitch/Roll).
  4. **Quality Intelligence**: Laplacian blur variance, photopic illumination, contrast, and composite 0–100 quality scoring.
  5. **Real-Time Vision**: Streamlined frame extraction with configurable detection intervals.
  6. **Secure AI Processing**: Privacy-first ephemeral processing with zero persistent biometric databases.
- **Session Telemetry**: Transparent browser session counters (`dv_images_analyzed`, `dv_videos_processed`, `dv_faces_detected`) without fabricated statistics.
- **Verified Pipeline Badges**: Real operational capabilities confirmed across Steps 1–5.

### `GET /image` — Image Face Analysis Studio
- **Upload Zone**: Drag-and-drop file uploader supporting JPEG, PNG, WebP, and BMP up to 15 MB.
- **Interactive Visualizer Canvas**: Renders detected faces with bounding boxes, 5 subpixel facial landmarks, interactive selection, and zoom/pan controls.
- **Multi-Face Tab Bar**: Switches between detected faces in crowd or multi-subject photographs.
- **Face Analysis Diagnostic Card**:
  - Detection confidence percentage.
  - Overall 0–100 quality score and qualitative grade (*Excellent*, *Good*, *Acceptable*, *Poor*).
  - Laplacian clarity & blur status.
  - Photopic brightness & exposure status.
  - Dynamic contrast & face pixel dimensions.
  - 3D head pose estimation indicators (Yaw, Pitch, Roll angles).
  - Expandable landmark coordinates table.

### `GET /video` — Video Multi-Face Tracking Studio
- **Upload & Options**: Ingests MP4, AVI, MOV, MKV, or WebM clips up to 500 MB. Configurable detection interval, analysis interval, and optional HUD rendering.
- **Asynchronous Job Management**: Submits jobs to `POST /api/v1/video/analyze` and polls status via `GET /api/v1/video/jobs/{job_id}` at 600 ms intervals.
- **Live Progress Card**: Displays genuine decoded frames, progress percentage, active faces, and unique track counts.
- **Video Result View**:
  - Embedded HTML5 video player streaming the annotated output artifact.
  - Video summary metrics (duration, resolution, native FPS, processing speed in FPS, total processing time).
  - Distinct visual track badges (`Track #1`, `Track #2`...) preserving session track IDs without person identification.

### `GET /camera` — Live Camera Studio
- WebRTC `navigator.mediaDevices.getUserMedia` video stream capture (1280×720).
- Hardware video initialization and teardown with clear roadmap notice for future WebSocket edge inference.

### `GET /analytics` — System Telemetry & Benchmarks
- Real-time hardware telemetry: active inference device (CPU SIMD / GPU), loaded neural engine (YuNet ONNX), and MOT algorithm (ByteTrack).
- Empirically measured stage latency benchmarks.

### `GET /settings` — Engine & System Preferences
- Interface theme selection (Dark AI Workspace vs. Light Mode) with instant DOM application and local persistence.
- Live backend connection diagnostics testing `GET /api/v1/health` and `GET /api/v1/ai/status` with round-trip ping latency.
- Tunable default detection confidence threshold (0.10–0.95), video detection interval, and analysis interval.

---

## 4. Component Hierarchy

```text
src/
├── app/
│   ├── layout.tsx                     # Root HTML, ThemeProvider, LayoutShell
│   ├── page.tsx                       # / landing page
│   ├── globals.css                    # CSS variables, animations, reduced-motion
│   ├── image/page.tsx                 # /image studio
│   ├── video/page.tsx                 # /video studio
│   ├── camera/page.tsx                # /camera studio
│   ├── analytics/page.tsx             # /analytics dashboard
│   └── settings/page.tsx              # /settings page
├── components/
│   ├── LayoutShell.tsx                # Responsive layout wrapper
│   ├── Header.tsx                     # Sticky bar, theme switch, system status indicator
│   ├── Sidebar.tsx                    # Collapsible navigation drawer
│   ├── common/
│   │   └── FileUploadZone.tsx         # Drag-and-drop, validation, size limits
│   ├── image/
│   │   ├── FaceVisualizerCanvas.tsx   # Interactive SVG/canvas face overlays
│   │   └── FaceAnalysisCard.tsx       # Multi-factor optical and pose diagnostics
│   └── video/
│       ├── VideoProcessingProgressCard.tsx # Frame counter, progress bar, cancel
│       └── VideoResultView.tsx        # Annotated player, metrics summary, track tags
├── context/
│   └── ThemeContext.tsx               # Dark/Light theme state and persistence
├── lib/
│   └── api/
│       ├── client.ts                  # Fetch wrapper, error parsing, timeout
│       ├── types.ts                   # Strongly-typed schemas matching backend
│       ├── image.ts                   # Image analysis API callers & normalizers
│       ├── video.ts                   # Video job submission, polling, cancellation
│       ├── system.ts                  # Health probing and status inspection
│       └── index.ts                   # Re-exported API interface
└── types/
    └── index.ts                       # Domain type re-exports
```

---

## 5. Accessibility & Design Standards

1. **Keyboard Navigation & Visible Focus Rings**:
   All buttons, links, and input elements implement `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan`.
2. **Screen Reader Support**:
   Interactive controls include descriptive `aria-label` attributes. Live updates (status indicators, upload errors) utilize `role="status"` and `role="alert"`.
3. **Contrast Compliance**:
   Colors meet WCAG 2.1 AA standards for text and interactive controls across both Dark and Light themes.
4. **Reduced Motion**:
   `@media (prefers-reduced-motion: reduce)` in `src/app/globals.css` turns off transitions and pulsing animations for users with motion sensitivity.
5. **Strict Truthfulness Policy**:
   Zero mock bounding boxes, zero fake progress percentages, and zero person identity naming. Track IDs are strictly presented as temporary session IDs.

---

## 6. Developer Commands

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server (port 3000)
npm run dev

# Run TypeScript type check
npx tsc --noEmit

# Run Next.js production build
npm run build

# Start production server
npm start
```

