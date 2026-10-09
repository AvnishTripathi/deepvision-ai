# DeepVision AI — Typed Configuration & Thresholds Reference (Step 19)

DeepVision AI utilizes **Pydantic Settings** for strongly typed, validated configuration. Settings can be overridden using environment variables or a `.env` file at the root of the project.

---

## 1. Feature Enablement Toggles

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `AI_ENABLE_LANDMARKS` | `bool` | `true` | Enables or disables 5-point facial landmark extraction and geometry validation. When disabled, landmarks processing is skipped (`landmarks=None`), and `landmark_time_ms` is `0.0`. |
| `AI_ENABLE_QUALITY_ANALYSIS` | `bool` | `true` | Enables or disables multi-factor face quality assessment (clarity, illumination, resolution, and composite scoring). When disabled, quality scoring is skipped and `quality_analysis_time_ms` is `0.0`. |
| `AI_ENABLE_POSE_ANALYSIS` | `bool` | `true` | Enables or disables 3D head pose estimation (Yaw, Pitch, Roll via `cv2.solvePnP`). Requires landmarks. When disabled, pose estimation is skipped (`pose=None`), and `pose_time_ms` is `0.0`. |

---

## 2. Configurable Optical & Quality Thresholds

| Setting | Type | Default | Valid Range | Methodology / Description |
| :--- | :--- | :--- | :--- | :--- |
| `FACE_BLUR_THRESHOLD` | `float` | `50.0` | $\ge 0.0$ | **Laplacian Variance Sharpness Cutoff (Step 7)**.<br>Faces with Laplacian variance $\ge 100.0$ are classified as `sharp`. Scores between `FACE_BLUR_THRESHOLD` (50.0) and 100.0 are `slightly_blurred`. Scores below `FACE_BLUR_THRESHOLD` are classified as `blurry`. |
| `FACE_BRIGHTNESS_MIN` | `float` | `60.0` | $[0.0, 255.0]$ | **Minimum Photopic Luminance (Step 8)**.<br>Calculated as mean gray luminance using ITU-R BT.601 ($Y = 0.299R + 0.587G + 0.114B$). Faces with mean luminance $< \text{FACE\_BRIGHTNESS\_MIN}$ are classified as `too_dark` (underexposed). |
| `FACE_BRIGHTNESS_MAX` | `float` | `200.0` | $[0.0, 255.0]$ | **Maximum Photopic Luminance (Step 8)**.<br>Faces with mean luminance $> \text{FACE\_BRIGHTNESS\_MAX}$ are classified as `too_bright` (overexposed/washed out). Luminance between `FACE_BRIGHTNESS_MIN` and `FACE_BRIGHTNESS_MAX` is classified as `acceptable`. |
| `FACE_QUALITY_THRESHOLD` | `float` | `45.0` | $[0.0, 100.0]$ | **Minimum Usable Composite Quality Score (Step 11 & 13)**.<br>Faces with overall quality score $< \text{FACE\_QUALITY\_THRESHOLD}$ are classified as `poor` and flagged `is_usable=False`. |

> [!NOTE]
> `FACE_BRIGHTNESS_MIN` must be strictly less than `FACE_BRIGHTNESS_MAX`. This is enforced automatically at startup via Pydantic model validation.

---

## 3. General AI Inference & Detector Settings

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `AI_DEVICE` | `str` | `auto` | Target inference hardware: `auto`, `cpu`, `gpu`, `cuda`, `mps`, `directml`. |
| `AI_ENABLE_GPU` | `bool` | `false` | Explicit flag to enable GPU acceleration if CUDA execution providers are present. |
| `AI_MODEL_NAME` | `str` | `scrfd_500m` | Name identifier of the active face detector model (e.g. `scrfd_500m`, `yunet`). |
| `AI_MODEL_VERSION` | `str` | `1.0.0` | Semantic version string for the active model checkpoint. |
| `AI_CONFIDENCE_THRESHOLD` | `float` | `0.50` | Minimum detector confidence threshold $[0.0, 1.0]$ for keeping candidate bounding boxes. |
| `AI_NMS_THRESHOLD` | `float` | `0.30` | Non-Maximum Suppression (NMS) intersection-over-union (IoU) overlap threshold. |
| `AI_MAX_IMAGE_SIZE_MB` | `int` | `10` | Hard cap on uploaded image file size in megabytes (enforces HTTP 413). |
| `AI_MAX_IMAGE_DIMENSION` | `int` | `4096` | Maximum allowed image width and height in pixels (mitigates decompression bombs). |

---

## 4. Security & Zero-Secret Exposure Guarantee

DeepVision AI enforces strict security hygiene regarding configuration telemetry:

1. **Volatile In-Memory Processing**: Uploaded images are never permanently written to disk during detection or analysis.
2. **Zero-Secret Exposure in Public APIs**:
   - `GET /api/v1/ai/status` exposes only safe, operational parameters via `settings.get_public_ai_config()`.
   - Sensitive infrastructure secrets—including `SECRET_KEY`, database connection strings (`DATABASE_URL` with credentials), and Redis connection strings (`REDIS_URL`)—are strictly filtered out and never serialized to client responses or application log files.
3. **Defense-in-Depth Validation**:
   - Out-of-range thresholds (e.g. negative blur thresholds, inverted brightness ranges, or quality scores $> 100$) trigger strict startup validation errors before any traffic is served.
