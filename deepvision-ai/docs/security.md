# DeepVision AI — Security Architecture & Vulnerability Audit Report

This document records the security architecture, threat model assessment, and vulnerability audit conducted for DeepVision AI (Step 7.23).

---

## 1. Security Architecture Summary

DeepVision AI implements defense-in-depth principles across its full application stack:

```text
 Client Request
       │
       ▼
 [Reverse Proxy / TLS Termination]
       │
       ▼
 [Security Middleware Layer]
   ├── TrustedHostMiddleware (Host Header Poisoning Defense)
   ├── Security Headers (HSTS, CSP, X-Frame-Options, Sniffing Defense)
   ├── Sliding Window Rate Limiter (Denial-of-Service Defense)
   └── CorrelationIdMiddleware (X-Request-ID Audit Tracking)
       │
       ▼
 [Validation & Sanitization Gate]
   ├── Strict MIME Type Enforcement
   ├── File Size / Dimension / Duration Guardrails
   ├── Magic Byte Video Stream Probe
   └── Path Traversal / Slug Sanitization
       │
       ▼
 [Decoupled Core Engines]
   ├── Zero Raw Disk Persistence for Images (In-Memory Processing)
   ├── Ephemeral Scoped Temporary Video Storage
   ├── Sandboxed Local/Object Storage Abstraction
   └── Non-Root Container Execution (UID 1000 / 1001)
```

---

## 2. Threat Vector Assessment & Audit Results

### 2.1 Hard-Coded Secrets & Credential Management
- **Status**: **PASS (Clean)**
- **Audit Findings**:
  - Zero hardcoded API keys, JWT signing keys, or database credentials exist in source code.
  - All sensitive parameters are resolved through Pydantic v2 `BaseSettings` reading environment variables or `.env`.
  - Production configurations reject default development keys and enforce secure 32+ byte cryptographically generated secret tokens.
  - Logging layer employs `SensitiveDataFilter` to automatically scrub passwords, tokens, API keys, and authorization headers.

### 2.2 File Upload & Unvalidated File Handling
- **Status**: **PASS (Enforced)**
- **Audit Findings**:
  - **MIME & Extension Validation**: Media uploads are restricted to certified types (`image/jpeg`, `image/png`, `image/webp` for images; `video/mp4`, `video/avi`, `video/mov`, `video/mkv` for video).
  - **Magic Byte & Stream Probing**: Video uploads undergo deep structural inspection using OpenCV video stream probing (`probe_and_validate_video`). Corrupted streams, truncated headers, or malformed containers are rejected immediately with descriptive client errors (`400 Bad Request` / `422 Unprocessable Content`).
  - **Resource Bounds**: Hard limits prevent decompression bombs:
    - Max image file size: 10 MB (configurable)
    - Max image dimensions: 4096 x 4096 px
    - Max video file size: 500 MB (configurable)
    - Max video duration: 600 seconds (10 minutes)
    - Max video resolution: 4K (4096 x 4096 px)
    - Max frame count: 18,000 frames

### 2.3 Path Traversal & File System Safety
- **Status**: **PASS (Protected)**
- **Audit Findings**:
  - Image uploads are decoded directly from memory buffers (`cv2.imdecode`) without ever touching the disk filesystem.
  - Video upload filenames are sanitized using `sanitize_video_filename()`, which strips path traversal sequences (`../`, `..\\`), directory separators, null bytes, and non-printable characters.
  - On-disk scratch files are generated using UUIDv4 filenames and Python's `tempfile` within isolated storage partitions (`storage/temp/`).

### 2.4 Command Injection & Arbitrary Code Execution
- **Status**: **PASS (Zero Risk)**
- **Audit Findings**:
  - Neither `subprocess`, `os.system`, `eval()`, nor `exec()` are utilized in backend application routing or AI engine processing.
  - All video processing, frame extraction, tracking, and rendering use native Python and OpenCV C++ bindings.

### 2.5 Cross-Origin Resource Sharing (CORS) & Host Validation
- **Status**: **PASS (Hardened)**
- **Audit Findings**:
  - FastAPI `CORSMiddleware` dynamically inspects configured origins.
  - If wildcard `*` is present in development, credential propagation (`allow_credentials`) is explicitly disabled to adhere to W3C specifications and prevent browser security exceptions.
  - In staging/production, `CORS_ORIGINS` is configured to explicit trusted domains (e.g., `https://app.deepvision.ai`).
  - `TrustedHostMiddleware` filters incoming `Host` headers to prevent HTTP Host Header Poisoning attacks.

### 2.6 Security Response Headers
- **Status**: **PASS (Active)**
- **Headers Enforced on Every Response**:
  - `X-Content-Type-Options: nosniff` (prevents MIME sniffing)
  - `X-Frame-Options: DENY` (clickjacking defense)
  - `X-XSS-Protection: 1; mode=block` (browser XSS filter)
  - `Referrer-Policy: strict-origin-when-cross-origin` (privacy-preserving referrers)
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains` (enforced automatically in production mode)

### 2.7 Denial of Service (DoS) & Concurrency Throttling
- **Status**: **PASS (Enforced)**
- **Audit Findings**:
  - **Endpoint Rate Limiting**: In-memory sliding-window rate limiter protects AI endpoints:
    - Image Detection: 60 requests/minute per client IP
    - Video Processing: 10 requests/minute per client IP
  - **Video Job Concurrency Limiter**: Maximum concurrent asynchronous video processing jobs bounded by `MAX_CONCURRENT_VIDEO_JOBS` (default: 4). Saturated queues return HTTP 429 Too Many Requests.

### 2.8 Error Handling & Traceback Masking
- **Status**: **PASS (Sanitized)**
- **Audit Findings**:
  - Global FastAPI exception handlers intercept unhandled server errors (`500 Internal Server Error`).
  - Raw Python tracebacks, file paths, database query syntax, and driver errors are never returned to clients.
  - Every error response returns a structured payload containing a unique correlation `request_id` for administrative server-side tracing.

### 2.9 Dependency Health Check
- **Status**: **PASS (Clean)**
- **Audit Findings**:
  - Automated dependency verification executed via `python -m pip check`.
  - Zero broken dependencies, missing requirements, or version conflicts detected.

---

## 3. Container & Deployment Security
- **Rootless Containers**: Both backend (`appuser`, UID 1000) and frontend (`nextjs`, UID 1001) run under unprivileged system accounts.
- **Minimal Image Footprint**: Multi-stage Docker builds based on `python:3.12-slim` and `node:20-alpine` eliminate unnecessary build tools from production images.
- **Read-Only / Segregated Volumes**: Storage partitions (`storage/temp`, `storage/media`, `storage/results`) are isolated from application source files.

