# DeepVision AI — Face Analysis Methodology & Technical Reference

This document provides a comprehensive technical reference for the **Face Analysis Subsystem** in DeepVision AI. It details the underlying mathematical formulations, optical algorithms, geometric models, transparent scoring criteria, and known engineering limitations.

---

## 1. Architectural Overview

The Face Analysis service operates downstream of the face detector:

```
                            ┌────────────────────────┐
                            │      Input Image       │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │     Face Detector      │
                            │    (SCRFD / YuNet)     │
                            └───────────┬────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
         ┌───────────────────────┐             ┌───────────────────────┐
         │ Bounding Box & Crop   │             │   5-Point Landmarks   │
         └───────────┬───────────┘             └───────────┬───────────┘
                     │                                     │
         ┌───────────┴───────────┐             ┌───────────┴───────────┐
         │  Optical Diagnostics  │             │ Geometric Diagnostics │
         │  ├── Blur (Laplacian) │             │  ├── IPD & Symmetry   │
         │  ├── Brightness (BT601│             │  └── 3D Head Pose     │
         │  ├── Contrast (StdDev)│             │      (solvePnP)       │
         │  └── Size & Area Ratio│             │                       │
         └───────────┬───────────┘             └───────────┬───────────┘
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │  Transparent Quality   │
                            │     Scoring Engine     │
                            │      (0 – 100)         │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │ Quality Classification │
                            │ (Excellent, Good, ...) │
                            └────────────────────────┘
```

---

## 2. Landmark Model & Extraction

### Model Architecture
- **Model**: YuNet (`face_detection_yunet_2023mar.onnx`) / SCRFD-500M.
- **Execution**: Evaluated in a single forward pass alongside bounding box regression via OpenCV DNN or ONNX Runtime.
- **Latency Impact**: $\approx 0\text{ ms}$ additional latency (regression heads are evaluated concurrently with box proposals).

### Landmark Format
The model natively extracts **5 canonical facial landmarks**:
1. `right_eye`: Center of right pupil/eye
2. `left_eye`: Center of left pupil/eye
3. `nose_tip`: Tip of the nose
4. `mouth_right`: Right corner of the mouth
5. `mouth_left`: Left corner of the mouth

#### Data Representations
- **List Schema**: `List[LandmarkPoint]` with sub-pixel native coordinates:
  ```json
  [
    {"name": "right_eye", "x": 234.5, "y": 182.1, "confidence": 0.98},
    {"name": "left_eye", "x": 312.8, "y": 181.4, "confidence": 0.98},
    {"name": "nose_tip", "x": 273.2, "y": 224.6, "confidence": 0.95},
    {"name": "mouth_right", "x": 245.1, "y": 268.3, "confidence": 0.92},
    {"name": "mouth_left", "x": 301.7, "y": 267.9, "confidence": 0.92}
  ]
  ```
- **Dictionary Schema**: Direct `{name: {x, y}}` mapping:
  ```json
  {
    "right_eye": {"x": 234.5, "y": 182.1},
    "left_eye": {"x": 312.8, "y": 181.4},
    "nose": {"x": 273.2, "y": 224.6},
    "right_mouth": {"x": 245.1, "y": 268.3},
    "left_mouth": {"x": 301.7, "y": 267.9}
  }
  ```

### Derived Geometric Diagnostics
From the 5 localized coordinates, the engine computes:
- **Inter-Pupillary Distance (IPD)**:
  $$\text{IPD} = \sqrt{(x_{\text{left\_eye}} - x_{\text{right\_eye}})^2 + (y_{\text{left\_eye}} - y_{\text{right\_eye}})^2}$$
- **Eye-Line Tilt Angle ($\theta$)**:
  $$\theta = \arctan2\left(y_{\text{left\_eye}} - y_{\text{right\_eye}}, \, x_{\text{left\_eye}} - x_{\text{right\_eye}}\right) \times \frac{180}{\pi}$$
- **Bilateral Facial Symmetry Ratio**: Ratio of distance from left eye to nose vs. right eye to nose ($1.0 = \text{perfect symmetry}$).

---

## 3. Optical Diagnostics Algorithms

### 3.1 Blur Detection Algorithm
Facial sharpness is measured using the **variance of the discrete Laplacian operator** on the grayscale face ROI:

$$\nabla^2 I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

The raw sharpness score corresponds to the spatial variance $\text{Var}(L)$:

$$\text{sharpness} = \frac{1}{N} \sum_{x, y} \left(\nabla^2 I(x, y) - \mu_L\right)^2$$

#### Status Classification
- **`sharp`**: $\text{sharpness} \ge 100.0$ (configurable via `AI_BLUR_SHARP_THRESHOLD`).
- **`slightly_blurred`**: $50.0 \le \text{sharpness} < 100.0$ (configurable via `FACE_BLUR_THRESHOLD`).
- **`blurry`**: $\text{sharpness} < 50.0$.

#### Continuous Score
Normalized to $S_{\text{blur}} \in [0.0, 1.0]$:
$$S_{\text{blur}} = \min\left(1.0, \, \max\left(0.0, \, \frac{\text{sharpness} - 15.0}{185.0}\right)\right)$$

---

### 3.2 Brightness Algorithm
Facial illumination is evaluated using **photopic luminance** based on the ITU-R BT.601 standard:

$$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$
$$\text{brightness} = \bar{Y} = \frac{1}{N} \sum_{x, y} Y(x, y)$$

#### Status Classification
- **`too_dark`**: $\bar{Y} < 60.0$ (`FACE_BRIGHTNESS_MIN`). Shadow clipping / underexposed.
- **`acceptable`**: $60.0 \le \bar{Y} \le 200.0$. Optimal photopic exposure.
- **`too_bright`**: $\bar{Y} > 200.0$ (`FACE_BRIGHTNESS_MAX`). Highlight clipping / overexposed.

---

### 3.3 Contrast Algorithm
Luminance dynamic range is assessed via the **standard deviation of photopic luminance**:

$$\text{contrast} = \sigma_Y = \sqrt{\frac{1}{N} \sum_{x, y} \left(Y(x, y) - \bar{Y}\right)^2}$$

#### Status Classification
- **`low_contrast`**: $\sigma_Y < 15.0$ (washed-out scene, fog, or flat diffuse illumination).
- **`acceptable`**: $15.0 \le \sigma_Y \le 75.0$ (balanced dynamic range preserving facial features).
- **`high_contrast`**: $\sigma_Y > 75.0$ (harsh directional lighting or strong side shadows).

---

### 3.4 Face Size & Scale Calculation
Evaluates biometric resolution sufficiency relative to original image canvas:

- **Dimensions**: $w = x_2 - x_1$, $h = y_2 - y_1$.
- **Area**: $\text{face\_area} = w \times h$.
- **Canvas Coverage Ratio**:
  $$\text{face\_area\_ratio} = \frac{w \times h}{\text{canvas\_width} \times \text{canvas\_height}}$$
- **Scale Category**:
  - `TINY`: $\max(w, h) < 32\text{px}$
  - `SMALL`: $32\text{px} \le \max(w, h) < 80\text{px}$
  - `MEDIUM`: $80\text{px} \le \max(w, h) < 200\text{px}$
  - `LARGE`: $\max(w, h) \ge 200\text{px}$
- **Biometric Sufficiency**: Requires $w \ge 32\text{px}$, $h \ge 32\text{px}$, and $\text{IPD} \ge 20\text{px}$.

---

## 4. 3D Head Pose Estimation Method

### Perspective-n-Point (solvePnP)
Head pose orientation is estimated by solving the **Perspective-n-Point (PnP)** problem between 2D localized landmarks and a canonical 3D human anthropometric facial model.

#### Canonical 3D Anthropometric Model Points (mm)
| Point | X (right) | Y (down) | Z (front) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Right Eye** | $-33.0$ | $-33.0$ | $-25.0$ | Right pupil center |
| **Left Eye** | $+33.0$ | $-33.0$ | $-25.0$ | Left pupil center |
| **Nose Tip** | $0.0$ | $0.0$ | $0.0$ | Canonical origin |
| **Mouth Right** | $-25.0$ | $+35.0$ | $-20.0$ | Right corner |
| **Mouth Left** | $+25.0$ | $+35.0$ | $-20.0$ | Left corner |

#### Camera Model & Solution
- Pinhole camera intrinsic matrix approximated from canvas dimensions:
  $$f_x = f_y = \max(W, H), \quad c_x = \frac{W}{2}, \quad c_y = \frac{H}{2}$$
- `cv2.solvePnP` computes rotation vector $\vec{r}$ and translation vector $\vec{t}$.
- `cv2.Rodrigues` converts $\vec{r}$ to rotation matrix $R \in \mathbb{R}^{3 \times 3}$.
- Euler angles extracted:
  - **Yaw ($\psi$)**: Horizontal rotation (negative = turned left, positive = turned right).
  - **Pitch ($\theta$)**: Vertical tilt (negative = tilted down, positive = tilted up).
  - **Roll ($\phi$)**: Coronal tilt (ear toward shoulder).

#### Frontality Score
Normalized continuous score $S_{\text{frontality}} \in [0.0, 1.0]$:
$$S_{\text{frontality}} = \max\left(0.0, \, 1.0 - \left(\frac{|\psi|}{45^\circ} \times 0.50 + \frac{|\theta|}{35^\circ} \times 0.30 + \frac{|\phi|}{30^\circ} \times 0.20\right)\right)$$

#### Unavailable State Handling
If reliable pose cannot be computed ($<5$ landmarks, missing canonical points, or $<3\text{px}$ inter-ocular distance), `pose` returns **`null`** rather than fabricated numbers.

> [!NOTE]
> Pose estimation values are engineering heuristics for facial orientation triage. They do **not** claim surgical, medical, or sub-degree biometric accuracy.

---

## 5. Transparent Face Quality Score Formula

DeepVision AI utilizes a fully explainable, transparent **0 – 100** scoring formula combining 5 measurable physical and geometric factors:

$$\text{Quality Score} = \text{Score}_{\text{base}}$$

```
Face Quality Score (0 – 100)
        │
        ├── Sharpness (25%)   ── Laplacian variance edge definition
        ├── Brightness (20%)  ── BT.601 photopic exposure suitability
        ├── Contrast (15%)    ── Dynamic range standard deviation
        ├── Face Size (20%)   ── Pixel resolution & scale sufficiency
        └── Landmark (20%)    ── Symmetry, availability & localization confidence
```

### Component Breakdown
$$\text{Score}_{\text{base}} = 0.25 \cdot F_{\text{sharpness}} + 0.20 \cdot F_{\text{brightness}} + 0.15 \cdot F_{\text{contrast}} + 0.20 \cdot F_{\text{size}} + 0.20 \cdot F_{\text{landmark}}$$

Where each sub-factor is strictly scaled to $[0.0, 100.0]$:
- $F_{\text{sharpness}} = 100 \times S_{\text{blur}}$ (penalized if categorized as `blurry`).
- $F_{\text{brightness}} = 100 \times S_{\text{brightness}}$ (peaked at mid-range $\approx 130$, penalized at $<60$ or $>200$).
- $F_{\text{contrast}} = \min(100.0, \, \max(0.0, \, f(\sigma_Y)))$ (optimal between $25$ and $65$).
- $F_{\text{size}} = f(\min(w, h))$ (penalized below $32\text{px}$, optimal $>150\text{px}$).
- $F_{\text{landmark}} = 100 \times (0.40 \cdot A_{\text{lm}} + 0.35 \cdot \text{symmetry} + 0.25 \cdot \text{confidence})$.

### Bottleneck Gating Rule
To prevent severely degraded images (e.g. pitch-black darkness, extreme motion blur, or sub-20px micro-faces) from achieving high scores due to compensation by other factors:

$$\text{critical\_min} = \min(F_{\text{sharpness}}, \, F_{\text{brightness}}, \, F_{\text{size}})$$
$$\text{if } \text{critical\_min} < 25.0 \implies \text{Quality Score} = \min(\text{Score}_{\text{base}}, \, \text{critical\_min} \times 1.80)$$

---

## 6. Classification Thresholds

The continuous 0 – 100 score maps to 4 standardized qualitative tiers:

| Tier | Quality Score | Description |
| :--- | :---: | :--- |
| **`excellent`** | $\ge 80.0$ | High-definition, sharp edges, optimal photopic exposure, strong facial symmetry. |
| **`good`** | $65.0 \dots 79.9$ | Minor imperfections; fully suitable for facial recognition and verification. |
| **`acceptable`** | $45.0 \dots 64.9$ | Marginal lighting or slight motion blur, but retains discernible geometry. |
| **`poor`** | $< 45.0$ | Severely degraded, blurry, harsh underexposure/overexposure, or microscopic resolution. |

Configurable globally via `FACE_QUALITY_THRESHOLD` (default: `45.0`).

---

## 7. Known Engineering Limitations

1. **Blur Metric vs. Skin Texture**:
   - Laplacian variance measures spatial edge frequency.
   - Diffuse lighting, youth skin, or heavy cosmetic makeup naturally lower variance without biometric degradation.
   - High ISO sensor noise or artificial grain can inflate variance.
2. **5-Point Landmark Pose Range**:
   - PnP estimation with 5 points is accurate up to $\approx \pm 45^\circ$ yaw.
   - Extreme profile views ($> 60^\circ$) where one eye is completely occluded cannot be solved accurately without dense 68-point or 468-point mesh models.
3. **Pinhole Camera Assumption**:
   - Focal length is estimated from canvas pixel dimensions ($f = \max(W, H)$) rather than calibrated lens parameters.
4. **Severe Occlusions**:
   - Heavy sunglasses, balaclavas, or medical masks occluding eyes or mouth degrade landmark localization certainty.
