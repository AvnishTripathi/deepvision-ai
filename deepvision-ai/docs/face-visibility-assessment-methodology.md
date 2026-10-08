# DeepVision AI — Face Visibility Assessment Methodology

## 1. Executive Summary & Objective

In production computer vision and biometric processing pipelines, an automated assessment of **Face Visibility** is required to decide whether a detected face has sufficient visual clarity, resolution, and structural integrity for downstream tasks (e.g., face recognition, attribute analysis, spoof detection, quality auditing).

This document specifies the methodology and scoring logic for the **Face Visibility Assessment** layer implemented in DeepVision AI (Step 10).

> [!NOTE]
> **Engineering Heuristic Disclaimer**:  
> Face visibility assessment in this system is an **engineering heuristic**, not an absolute or universal ground-truth measurement. Optical factors such as atmospheric haze, uneven artistic lighting, facial accessories, or atypical morphology can alter individual factor values without necessarily compromising human perceptibility. The heuristic provides an explainable, deterministic baseline for automated system gating and triage.

---

## 2. Multi-Dimensional Visibility Factors

The visibility assessment synthesizes five distinct visual dimensions into normalized factors $S_i \in [0.0, 1.0]$:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Face Visibility Assessment                            │
├──────────────┬──────────────┬──────────────┬────────────────┬───────────────┤
│  Face Size   │  Sharpness   │  Brightness  │ Landmark Avail │ Landmark Conf │
│ (Resolution) │ (Laplacian)  │ (BT.601 Y)   │  (5 Points)    │  (Confidence) │
│   [0.0-1.0]  │  [0.0-1.0]   │  [0.0-1.0]   │   [0.0-1.0]    │   [0.0-1.0]   │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┴───────┬───────┘
       │ (25%)        │ (25%)        │ (25%)          │ (15%)         │ (10%)
       └──────────────┼──────────────┼────────────────┴───────────────┘
                      ▼
            Weighted Baseline Score
                      │
                      ▼
          Critical Bottleneck Gating
      (Limits score if min factor < 0.35)
                      │
                      ▼
            Composite Visibility
      [excellent | good | fair | poor]
```

### 2.1. Face Size Factor ($S_{\text{size}}$)
- **Measurement**: Minimum face dimension $D_{\min} = \min(\text{width}, \text{height})$ in native image pixels.
- **Biometric Rationale**: Faces smaller than 32px lack sufficient Nyquist sampling density for reliable biometric feature extraction. Faces above 80px provide distinct feature separation, while faces above 200px provide high-definition texture.
- **Mathematical Formula**:
  $$S_{\text{size}} = \begin{cases}
  0.0, & \text{if } D_{\min} \le 0 \\
  \max\left(0.10, \frac{D_{\min}}{32} \times 0.40\right), & \text{if } D_{\min} < 32\text{ px (TINY)} \\
  0.40 + \frac{D_{\min} - 32}{48} \times 0.35, & \text{if } 32 \le D_{\min} < 80\text{ px (SMALL)} \\
  0.75 + \frac{D_{\min} - 80}{120} \times 0.20, & \text{if } 80 \le D_{\min} < 200\text{ px (MEDIUM)} \\
  \min\left(1.0, 0.95 + \frac{D_{\min} - 200}{400} \times 0.05\right), & \text{if } D_{\min} \ge 200\text{ px (LARGE)}
  \end{cases}$$

### 2.2. Blur & Sharpness Factor ($S_{\text{blur}}$)
- **Measurement**: High-frequency spatial variance computed via the discrete 2D Laplacian operator:
  $$\text{Var}(\Delta) = \frac{1}{HW} \sum_{x,y} \left( \nabla^2 I(x, y) - \mu_{\nabla^2} \right)^2$$
- **Mapping**: Derived from `ClarityMetrics.blur_score` and `blur_status`:
  - If `blur_status == BlurStatus.BLURRY`: $S_{\text{blur}} \le 0.35$ (hard limit for motion/optical blur).
  - If `blur_status == BlurStatus.SLIGHTLY_BLURRED`: $S_{\text{blur}} \in [0.40, 0.74]$.
  - If `blur_status == BlurStatus.SHARP`: $S_{\text{blur}} \ge 0.75$.

### 2.3. Brightness & Illumination Factor ($S_{\text{brightness}}$)
- **Measurement**: ITU-R BT.601 perceptual luminance $Y = 0.299 R + 0.587 G + 0.114 B$.
- **Mapping**: Derived from `IlluminationMetrics.brightness_score` and `brightness_status`:
  - If `brightness_status in (TOO_DARK, TOO_BRIGHT)`: $S_{\text{brightness}} \le 0.50$ (penalizes shadow crushing or specular clipping).
  - If `brightness_status == ACCEPTABLE` ($60.0 \le \mu_Y \le 200.0$): $S_{\text{brightness}} \in [0.70, 1.00]$, peaking at the optimal midpoint ($\mu_Y = 130.0$).

### 2.4. Landmark Availability Factor ($S_{\text{lm\_avail}}$)
- **Measurement**: Number of verified facial landmark coordinates $N_{\text{detected}}$ relative to canonical 5-point layout (`left_eye`, `right_eye`, `nose`, `left_mouth`, `right_mouth`).
- **Mapping**:
  $$S_{\text{lm\_avail}} = \begin{cases}
  0.20, & \text{if } N_{\text{detected}} = 0 \text{ (detection only, facial structure unverified)} \\
  \max\left(0.20, \min\left(1.0, \frac{N_{\text{detected}}}{5.0}\right)\right), & \text{if } 0 < N_{\text{detected}} < 5 \\
  1.0, & \text{if } N_{\text{detected}} \ge 5
  \end{cases}$$

### 2.5. Landmark Confidence Factor ($S_{\text{lm\_conf}}$)
- **Measurement**: Mean confidence score $c_i \in [0.0, 1.0]$ across all detected landmarks.
- **Fallback**:
  - If the active detector returns explicit landmark confidences: $S_{\text{lm\_conf}} = \frac{1}{K} \sum_{i=1}^K c_i$.
  - If landmarks are present but lack individual confidence scores (e.g., standard coordinate outputs): falls back to the face bounding box detection confidence $c_{\text{det}}$.
  - If no landmarks were detected: falls back to $\min(0.50, c_{\text{det}} \times 0.50)$.

---

## 3. Aggregation & Critical Bottleneck Gating

### 3.1. Linear Weighted Baseline
The continuous baseline score balances spatial resolution, focus, exposure, and structural geometry:
$$\text{Score}_{\text{base}} = 0.25 \cdot S_{\text{size}} + 0.25 \cdot S_{\text{blur}} + 0.25 \cdot S_{\text{brightness}} + 0.15 \cdot S_{\text{lm\_avail}} + 0.10 \cdot S_{\text{lm\_conf}}$$

### 3.2. Critical Bottleneck Gating (Veto Rule)
A face cannot be deemed "good" or "excellent" if any single fundamental visual property is critically degraded:
- Pitch-black illumination ($\mu_Y < 20$) renders facial contours invisible even if resolution is $1000\text{px}$.
- Extreme motion blur obliterates biometric identity markers.
- Microscopic faces ($< 16\text{px}$) cannot be inspected reliably.

We define the critical minimum:
$$M_{\text{crit}} = \min(S_{\text{size}}, S_{\text{blur}}, S_{\text{brightness}})$$

When $M_{\text{crit}} < 0.35$, the composite score is bounded:
$$\text{Score}_{\text{final}} = \min\left(\text{Score}_{\text{base}}, M_{\text{crit}} \times 1.60\right)$$
$$\text{visibility\_score} = \text{round}\left(\min(1.0, \max(0.0, \text{Score}_{\text{final}})), 3\right)$$

---

## 4. Qualitative Visibility Rating Mapping

The final continuous `visibility_score` is mapped into four qualitative categories:

| Status | Score Threshold | Critical Min Requirement | Landmark Requirement | Typical Visual Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **`excellent`** | $\ge 0.82$ | $M_{\text{crit}} \ge 0.60$ | $S_{\text{lm\_avail}} \ge 0.80$ | High resolution, sharp focus, balanced exposure, complete 5-point landmarks. |
| **`good`** | $\ge 0.65$ | $M_{\text{crit}} \ge 0.38$ | — | Acceptable resolution ($> 50\text{px}$), minor blur or slight exposure variance, landmarks localized. |
| **`fair`** | $\ge 0.45$ | — | — | Low resolution ($32\text{px}-50\text{px}$), moderate blur, or uneven lighting; usable for detection but degraded. |
| **`poor`** | $< 0.45$ | — | — | Microscopic scale ($< 32\text{px}$), severe motion blur, pitch black or blown-out white, missing landmarks. |

---

## 5. Configuration & Overrides

All threshold boundaries are dynamically configurable without code modifications:

| Setting Key | Environment Variable | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `AI_VISIBILITY_EXCELLENT_THRESHOLD` | `AI_VISIBILITY_EXCELLENT_THRESHOLD` | `0.82` | Cutoff for `excellent` visibility status |
| `AI_VISIBILITY_GOOD_THRESHOLD` | `AI_VISIBILITY_GOOD_THRESHOLD` | `0.65` | Cutoff for `good` visibility status |
| `AI_VISIBILITY_FAIR_THRESHOLD` | `AI_VISIBILITY_FAIR_THRESHOLD` | `0.45` | Cutoff for `fair` visibility status |

Furthermore, the calculation function `calculate_face_visibility()` accepts explicit runtime override parameters:
```python
assessment = calculate_face_visibility(
    face_crop_rgb=crop,
    bbox=bbox,
    image_shape=(1080, 1920),
    excellent_threshold=0.85,
    good_threshold=0.70,
    fair_threshold=0.50,
)
```

---

## 6. Output Schema Structure

The assessment is returned as part of each analyzed face in `StructuredFaceAnalysis`:

```json
{
  "visibility": "excellent",
  "visibility_assessment": {
    "visibility": "excellent",
    "visibility_score": 0.895,
    "size_factor": 0.95,
    "blur_factor": 0.88,
    "brightness_factor": 0.92,
    "landmark_availability_factor": 1.0,
    "landmark_confidence_factor": 0.98,
    "details": "Optimal facial visibility across all metrics"
  }
}
```

For degraded faces, `details` identifies the primary limiting factor:
- `"Significant blur detected"`
- `"Suboptimal illumination (underexposed or overexposed)"`
- `"Low face resolution / tiny scale"`
- `"Incomplete facial landmarks"`
