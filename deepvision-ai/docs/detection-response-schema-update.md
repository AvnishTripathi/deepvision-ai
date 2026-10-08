# DeepVision AI — Detection Response Schema Specification (Step 14)

## 1. Overview & Architectural Context

In **Step 14**, the detection response of DeepVision AI (`POST /api/v1/ai/detect`) is extended to integrate all physical, optical, and geometric analytics developed across Steps 4 through 13 into a single unified JSON response payload:

- **Structured Facial Landmarks** (`left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`)
- **Transparent Quality Analytics** (`score`, `classification`, `blur_score`, `blur_status`, `brightness_score`, `brightness_status`, `contrast_score`, `contrast_status`, `face_area_ratio`, `visibility`)
- **3D Head Pose Orientation** (`yaw`, `pitch`, `roll`)

> [!IMPORTANT]
> **Strict Backward Compatibility**:  
> All existing fields (`success`, `model`, `device`, `image`, `processing_time_ms`, `face_count`, `faces`, `timing`, `request_id`) are preserved.  
> Legacy consumers querying quality attributes (`overall_quality`, `sharpness`, `brightness`, `contrast`, `is_usable`) or landmark representations (`landmarks_list`, `landmarks_dict`) remain 100% operational without breakage.

---

## 2. Standardized JSON Response Structure

```json
{
  "success": true,
  "model": "scrfd_500m",
  "device": "CPU",
  "image": {
    "width": 1920,
    "height": 1080
  },
  "processing_time_ms": 45.2,
  "face_count": 1,
  "faces": [
    {
      "face_id": null,
      "bbox": {
        "x1": 100.0,
        "y1": 120.0,
        "x2": 350.0,
        "y2": 380.0
      },
      "confidence": 0.97,
      "landmarks": {
        "left_eye": {
          "x": 150.0,
          "y": 180.0
        },
        "right_eye": {
          "x": 220.0,
          "y": 180.0
        },
        "nose": {
          "x": 185.0,
          "y": 220.0
        },
        "left_mouth": {
          "x": 160.0,
          "y": 260.0
        },
        "right_mouth": {
          "x": 210.0,
          "y": 260.0
        }
      },
      "quality": {
        "score": 91.4,
        "classification": "excellent",
        "blur_score": 142.5,
        "blur_status": "sharp",
        "brightness_score": 126.2,
        "brightness_status": "acceptable",
        "contrast_score": 51.4,
        "contrast_status": "acceptable",
        "face_area_ratio": 0.08,
        "visibility": "excellent"
      },
      "pose": {
        "yaw": 3.1,
        "pitch": -1.2,
        "roll": 0.5
      }
    }
  ]
}
```

---

## 3. Field Definitions & Value Specifications

### 3.1. Top-Level Envelope
| Field | Type | Description |
| :--- | :--- | :--- |
| `success` | `bool` | `true` if detection completed successfully. |
| `model` | `string` | Identifier of active detector (e.g. `"scrfd_500m"`). |
| `device` | `string` | Target hardware execution platform (`"CPU"`, `"GPU"`). |
| `image.width` | `integer` | Pixel width of uploaded image canvas. |
| `image.height` | `integer` | Pixel height of uploaded image canvas. |
| `processing_time_ms` | `float` | Total end-to-end processing latency in milliseconds. |
| `face_count` | `integer` | Total number of detected faces exceeding confidence threshold. |
| `faces` | `array[object]` | List of detected face records. |
| `timing` *(optional)* | `object` | Granular stage breakdown (`preprocessing`, `inference`, `postprocessing`). |
| `request_id` *(optional)* | `string` | Unique request trace identifier. |

---

### 3.2. Face Item (`faces[]`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `face_id` | `string` \| `integer` \| `null` | Tracking/identity identifier (`null` for single image detection). |
| `bbox` | `object` | Bounding box coordinates `{x1, y1, x2, y2}` in original canvas space. |
| `confidence` | `float` | Detection score bounded in $[0.0, 1.0]$. |
| `landmarks` | `object` \| `null` | Structured map `{left_eye, right_eye, nose, left_mouth, right_mouth}` with `{x, y}` coordinates in pixels. |
| `quality` | `object` | Detailed visual quality metrics (see table below). |
| `pose` | `object` \| `null` | 3D head pose orientation angles `{yaw, pitch, roll}` in degrees ($^\circ$). |

---

### 3.3. Quality Diagnostics (`faces[].quality`)
| Property | Type | Metric / Calculation | Allowed Values / Ranges |
| :--- | :---: | :--- | :--- |
| `score` | `float` | Multi-factor transparent composite quality score | Continuous $[0.0, 100.0]$ |
| `classification` | `string` | Operational utility tier (Step 13) | `"excellent"`, `"good"`, `"acceptable"`, `"poor"` |
| `blur_score` | `float` | High-frequency Laplacian variance $\sigma_{\Delta}^2$ | $\ge 0.0$ (e.g. $142.5$) |
| `blur_status` | `string` | Optical sharpness categorization (Step 7) | `"sharp"`, `"slightly_blurred"`, `"blurry"` |
| `brightness_score` | `float` | Mean photopic luminance $\mu_Y$ (ITU-R BT.601) | Continuous $[0.0, 255.0]$ |
| `brightness_status` | `string` | Photopic exposure categorization (Step 8) | `"too_dark"`, `"acceptable"`, `"too_bright"` |
| `contrast_score` | `float` | Luminance standard deviation $\sigma_Y$ | Continuous $[0.0, 128.0]$ |
| `contrast_status` | `string` | Dynamic range classification | `"acceptable"`, `"low_contrast"`, `"high_contrast"` |
| `face_area_ratio` | `float` | Ratio of face bounding box area to total image canvas area | Continuous $[0.0, 1.0]$ |
| `visibility` | `string` | Overall observability rating (Step 10) | `"excellent"`, `"good"`, `"fair"`, `"poor"` |

---

### 3.4. Head Pose (`faces[].pose`)
Estimated via OpenCV `solvePnP` against canonical 3D craniofacial anthropometry:
- `yaw`: Horizontal rotation around vertical axis ($^\circ$)
- `pitch`: Vertical nod tilt around transverse axis ($^\circ$)
- `roll`: Sideways tilt around optical axis ($^\circ$)

> [!NOTE]
> When landmarks are incomplete, degenerate, or perspective convergence fails, `pose` is set to `null` to prevent emitting misleading fake zero angles.

---

## 4. Backward Compatibility Contract

Existing client applications expecting legacy properties can seamlessly continue consuming:
- `face.quality.overall_quality`: Normalized $[0.0, 1.0]$ floating-point score ($=\text{score} / 100.0$).
- `face.quality.sharpness`: Identical to `blur_score`.
- `face.quality.brightness`: Identical to `brightness_score`.
- `face.quality.contrast`: Identical to `contrast_score`.
- `face.quality.is_usable`: Boolean gate ($\text{score} \ge 30.0$).
- `face.landmarks_list`: Raw array representation of landmarks `[{"name": "...", "x": ..., "y": ...}]`.
- `face.landmarks_dict`: Structured map matching `face.landmarks`.
