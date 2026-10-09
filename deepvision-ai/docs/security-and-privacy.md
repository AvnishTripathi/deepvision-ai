# DeepVision AI — Security & Privacy Architecture (Step 22)

## 1. Overview & Scope

DeepVision AI is strictly an **image-level computer vision analysis engine**. The platform evaluates physical scene optics, facial geometry, illumination sufficiency, sharpness variance, and 3D head pose orientation on consented image uploads.

This stage does **NOT** perform biometric identification, facial recognition, surveillance, or person tracking.

```
                              ┌──────────────────────────────────┐
                              │     Consented User Upload        │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ 9 Ingestion Security Controls    │
                              │ (MIME, Magic Bytes, 10MB Cap)    │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Volatile In-Memory Processing    │
                              │ ├── Bounding Box (SCRFD/YuNet)   │
                              │ ├── 5-Point Landmarks            │
                              │ ├── Optics (Blur, Brightness)    │
                              │ └── 3D Head Pose Orientation     │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │ Ephemeral JSON Response          │
                              │ (Discard all memory & images)    │
                              └──────────────────────────────────┘
                                               │
        ❌ NO Identity           ❌ NO Biometric Profile        ❌ NO Disk Storage
        ❌ NO Vector Database    ❌ NO Face Scraping            ❌ NO Surveillance
```

---

## 2. Core Privacy Guarantees (The 5 Explicit "Do NOT" Mandates)

### 2.1 DO NOT Identify People
- **Policy**: The service does **not** identify or attempt to identify human individuals.
- **Implementation**:
  - The model outputs only bounding boxes, landmark coordinates, and optical metrics.
  - The API schemas ([`DetectedFaceItem`](file:///c:/Users/abhis/Music/Deep%20Face%20Detection/deepvision-ai/backend/app/schemas/ai.py#L320-L345) and [`StructuredFaceAnalysis`](file:///c:/Users/abhis/Music/Deep%20Face%20Detection/deepvision-ai/backend/app/ai/analysis/schemas.py#L436-L478)) contain no `identity`, `person_name`, `user_id`, `social_security_number`, or individual identifiers.
  - The `face_id` field is strictly an integer rank (e.g. `0`, `1`, `2`) indicating face order within a single image frame.

### 2.2 DO NOT Match Faces Against Databases
- **Policy**: The service does **not** compare, query, or match detected faces against any gallery, watchlist, or identity database.
- **Implementation**:
  - No vector databases (FAISS, Milvus, Qdrant, pgvector) are integrated into the detection/analysis pipeline.
  - No cosine similarity, Euclidean embedding distance, or 1:1 / 1:N biometric matching logic exists in the analysis pipeline.

### 2.3 DO NOT Scrape Faces
- **Policy**: The service does **not** crawl, scrape, aggregate, or harvest facial imagery from web pages, social media platforms, or external repositories.
- **Implementation**:
  - No automated scrapers, web spiders, or background crawlers exist in the codebase.
  - Processing occurs solely on user-initiated, explicitly uploaded payloads.

### 2.4 DO NOT Create Biometric Profiles
- **Policy**: The service does **not** construct, store, or serialize persistent biometric profiles or behavioral identity templates.
- **Implementation**:
  - No 128-D, 512-D, or high-dimensional face recognition embeddings (FaceNet, ArcFace, CosFace) are computed, saved, or indexed.
  - Landmarks and pose estimates are calculated strictly as transient geometric diagnostics for image quality and orientation triage.

### 2.5 DO NOT Perform Covert Surveillance
- **Policy**: The platform is explicitly prohibited from conducting passive, hidden, or continuous surveillance monitoring.
- **Implementation**:
  - The architecture is stateless and request-driven: the client sends an image, receives the analysis JSON, and all buffers are immediately released.
  - No persistent background camera feeds, RTSP interceptors, or covert capture daemons are operated.

---

## 3. Data Retention & Ephemeral Storage Policy

### Zero Permanent Disk Storage of Biometric Imagery
- Uploaded imagery processed via `/api/v1/ai/detect` and `/api/v1/ai/analyze` is ingested **strictly in volatile system memory (RAM)**.
- Image byte streams are decoded directly into NumPy RGB arrays and released upon completion of request serialization.
- No raw images, cropped face ROIs, or intermediate visual representations are written to disk, databases, or S3/cloud buckets.
- If isolated temporary files are required by external libraries (e.g., specific CLI tools), they use [`secure_temporary_image`](file:///c:/Users/abhis/Music/Deep%20Face%20Detection/deepvision-ai/backend/app/core/image_security.py#L210-L245) with restrictive POSIX/Windows permissions (`0o600`) and automatic `try...finally` deletion.

---

## 4. Ingestion Security Controls

DeepVision AI enforces 9 layered defense controls on all ingested images:

| Control | Mechanism | Threat Mitigated |
| :--- | :--- | :--- |
| **1. MIME Validation** | Whitelist inspection (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`) | Arbitrary file upload |
| **2. Magic Bytes Check** | Binary header validation (SOI `\xFF\xD8\xFF`, PNG `\x89PNG\r\n\x1a\n`, RIFF/WEBP) | MIME-type spoofing / polyglot payloads |
| **3. Hard Payload Cap** | Stream reading aborts if payload exceeds `AI_MAX_IMAGE_SIZE_MB` (default 10 MB) | Memory exhaustion / DoS attacks |
| **4. Dimension Limits** | Rejects images exceeding `AI_MAX_IMAGE_DIMENSION` (4096 px) | Decompression bomb / pixel flood attacks |
| **5. Execution Prevention** | Rejection of executable binaries (`.exe`, `.sh`, `.php`, `.py`) | Remote code execution |
| **6. Filename Sanitization** | Stripping path traversals (`../`), null bytes (`\x00`), and special symbols | Directory traversal / arbitrary overwrite |
| **7. Ephemeral Isolation** | Memory-backed streams and strictly scoped temp contexts | Residual data leakage |
| **8. Zero Disk Persistence** | Pure in-memory array decoding with automatic garbage collection | Unconsented image retention |
| **9. Zero Raw Data Logging** | Structured logs record only request ID, dimensions, face count, and latency | Sensitive biometric data leakage in logs |

---

## 5. Summary Compliance Matrix

| Requirement | Status | Verification |
| :--- | :---: | :--- |
| **Identify people** | ❌ **PROHIBITED** | Verified: Schemas contain zero identity fields |
| **Match faces against databases** | ❌ **PROHIBITED** | Verified: Zero vector search or gallery matching |
| **Scrape faces** | ❌ **PROHIBITED** | Verified: Zero scraping tools or web crawlers |
| **Create biometric profiles** | ❌ **PROHIBITED** | Verified: Zero biometric templates or embedding persistence |
| **Perform covert surveillance** | ❌ **PROHIBITED** | Verified: Fully stateless, synchronous API request/response |
| **Permanent biometric storage** | ❌ **PROHIBITED** | Verified: Pure in-memory decoding; zero disk persistence |
| **Security controls active** | ✅ **ENFORCED** | Verified: Full ingestion security test suite passing |
