# DeepVision AI — Face Quality Score Methodology (0 - 100)

## 1. Overview & Architectural Philosophy

The **Face Quality Score** in DeepVision AI provides a standardized, explainable metric scaled from **0 to 100** evaluating the visual and biometric suitability of detected face regions.

> [!IMPORTANT]
> **No Arbitrary AI Numbers**:  
> In contrast to opaque deep learning "black box" quality estimators that output uninterpretable probability vectors, DeepVision AI computes its quality score using **transparent, deterministic, and physically grounded computer vision algorithms**. Every point in the 0 - 100 score can be traced directly to physical optical measurements or geometric facial invariants.

---

## 2. Factor Architecture

The overall score synthesizes five foundational optical and geometric factors:

```
Face Quality Score (0 - 100)
        │
        ├── Sharpness        (25% weight)
        ├── Brightness       (20% weight)
        ├── Contrast         (15% weight)
        ├── Face Size        (20% weight)
        └── Landmark Quality (20% weight)
```

| Factor | Weight | Measurable Property | Optical / Physical Justification |
| :--- | :---: | :--- | :--- |
| **Sharpness** | **25%** | Laplacian variance ($\sigma_{\Delta}^2$) | High-frequency edge definition distinguishing in-focus facial features from motion or defocus blur. |
| **Brightness** | **20%** | ITU-R BT.601 mean luminance ($\mu_Y$) | Sensor exposure level avoiding shadow crushing ($< 60$) and specular blowout ($> 200$). |
| **Contrast** | **15%** | Luminance standard deviation ($\sigma_Y$) | Dynamic range distinguishing textured facial contours from washed-out or flat illumination. |
| **Face Size** | **20%** | Minimum pixel dimension $D_{\min}$ | Resolution density ensuring sufficient spatial samples (Nyquist limit) for feature extraction. |
| **Landmark Quality** | **20%** | Geometric symmetry & completeness | Bilateral facial symmetry ratio and landmark localization certainty. |

---

## 3. Mathematical Formulation of Measurable Factors

All five sub-scores are independently mapped into the continuous range $[0.0, 100.0]$.

### 3.1. Sharpness Factor ($F_{\text{sharpness}}$)
- **Raw Metric**: Discrete Laplacian variance on grayscale face crop $Y$:
  $$\sigma_{\Delta}^2 = \frac{1}{HW} \sum_{x,y} \left( \nabla^2 Y(x, y) - \mu_{\nabla^2} \right)^2$$
- **Mapping**:
  $$F_{\text{sharpness}} = 100.0 \times \min\left(1.0, \max\left(0.0, \frac{\sigma_{\Delta}^2 - 15.0}{185.0}\right)\right)$$
  - Blurry status ($\sigma_{\Delta}^2 < 50.0$): capped at $35.0$.
  - Slightly blurred ($50.0 \le \sigma_{\Delta}^2 < 100.0$): bounded between $40.0$ and $74.0$.
  - Sharp ($\sigma_{\Delta}^2 \ge 100.0$): scaled smoothly from $75.0$ to $100.0$.

### 3.2. Brightness Factor ($F_{\text{brightness}}$)
- **Raw Metric**: ITU-R BT.601 photopic luminance average:
  $$\mu_Y = \frac{1}{N} \sum_{i=1}^N \left( 0.299 R_i + 0.587 G_i + 0.114 B_i \right)$$
- **Mapping**: Evaluated against configurable shadow ($T_{\min} = 60.0$) and highlight ($T_{\max} = 200.0$) thresholds, centered at optimal midpoint $\mu_{\text{mid}} = 130.0$:
  $$F_{\text{brightness}} = 100.0 \times \begin{cases}
  \max\left(0.0, \frac{\mu_Y}{T_{\min}} \times 0.70\right), & \text{if } \mu_Y < T_{\min}\text{ (too dark)} \\
  \max\left(0.70, 1.0 - 0.30 \times \frac{|\mu_Y - \mu_{\text{mid}}|}{(T_{\max} - T_{\min}) / 2}\right), & \text{if } T_{\min} \le \mu_Y \le T_{\max}\text{ (acceptable)} \\
  \max\left(0.0, \frac{255.0 - \mu_Y}{255.0 - T_{\max}} \times 0.70\right), & \text{if } \mu_Y > T_{\max}\text{ (too bright)}
  \end{cases}$$

### 3.3. Contrast Factor ($F_{\text{contrast}}$)
- **Raw Metric**: Sample standard deviation of luminance:
  $$\sigma_Y = \sqrt{\frac{1}{N} \sum_{i=1}^N (Y_i - \mu_Y)^2}$$
- **Mapping**:
  $$F_{\text{contrast}} = 100.0 \times \begin{cases}
  \max\left(0.0, \frac{\sigma_Y}{15.0} \times 0.50\right), & \text{if } \sigma_Y < 15.0\text{ (low contrast)} \\
  0.50 + \frac{\sigma_Y - 15.0}{25.0} \times 0.50, & \text{if } 15.0 \le \sigma_Y < 40.0 \\
  1.00, & \text{if } 40.0 \le \sigma_Y \le 65.0\text{ (optimal)} \\
  \max\left(0.40, 1.0 - \frac{\sigma_Y - 65.0}{35.0} \times 0.30\right), & \text{if } \sigma_Y > 65.0\text{ (harsh shadows)}
  \end{cases}$$

### 3.4. Face Size Factor ($F_{\text{size}}$)
- **Raw Metric**: Minimum bounding box dimension $D_{\min} = \min(\text{width}, \text{height})$ in pixels:
  $$F_{\text{size}} = 100.0 \times \begin{cases}
  0.0, & \text{if } D_{\min} \le 0 \\
  \max\left(0.05, \frac{D_{\min}}{32.0} \times 0.40\right), & \text{if } D_{\min} < 32\text{ px (TINY)} \\
  0.40 + \frac{D_{\min} - 32.0}{48.0} \times 0.35, & \text{if } 32 \le D_{\min} < 80\text{ px (SMALL)} \\
  0.75 + \frac{D_{\min} - 80.0}{120.0} \times 0.20, & \text{if } 80 \le D_{\min} < 200\text{ px (MEDIUM)} \\
  \min\left(1.0, 0.95 + \frac{D_{\min} - 200.0}{400.0} \times 0.05\right), & \text{if } D_{\min} \ge 200\text{ px (LARGE)}
  \end{cases}$$

### 3.5. Landmark Quality Factor ($F_{\text{landmark}}$)
- **Raw Metrics**:
  - Landmark Completeness $A_{\text{lm}} = \min(1.0, \frac{N_{\text{landmarks}}}{5.0})$ ($0.20$ if missing).
  - Bilateral Facial Symmetry $S_{\text{sym}} \in [0.0, 1.0]$: Ratio of Euclidean distances between nose tip and left/right eyes.
  - Localization Certainty $C_{\text{lm}} \in [0.0, 1.0]$: Mean confidence score of localized keypoints.
- **Mapping**:
  $$F_{\text{landmark}} = 100.0 \times \left(0.40 \cdot A_{\text{lm}} + 0.35 \cdot S_{\text{sym}} + 0.25 \cdot C_{\text{lm}}\right)$$

---

## 4. Aggregation Formula & Critical Gating

### 4.1. Linear Weighted Baseline
$$\text{Score}_{\text{base}} = 0.25 \cdot F_{\text{sharpness}} + 0.20 \cdot F_{\text{brightness}} + 0.15 \cdot F_{\text{contrast}} + 0.20 \cdot F_{\text{size}} + 0.20 \cdot F_{\text{landmark}}$$

### 4.2. Bottleneck Gating (Veto Penalty)
If any fundamental visual property is critically degraded (e.g. pitch black, total motion blur, or sub-32px tiny face), the overall quality cannot be artificially inflated:
$$M_{\text{crit}} = \min(F_{\text{sharpness}}, F_{\text{brightness}}, F_{\text{size}})$$

$$\text{quality\_score} = \begin{cases}
\min\left(\text{Score}_{\text{base}}, M_{\text{crit}} \times 1.80\right), & \text{if } M_{\text{crit}} < 25.0 \\
\text{Score}_{\text{base}}, & \text{otherwise}
\end{cases}$$

Clamped strictly to $[0.0, 100.0]$ and rounded to one decimal place.

---

## 5. Qualitative Grades

The continuous $0 - 100$ score maps to standardized biometric quality grades:

| Grade | Score Range (0 - 100) | Normalized (0.0 - 1.0) | Production Utility |
| :--- | :---: | :---: | :--- |
| **`EXCELLENT`** | $\ge 85.0$ | $\ge 0.85$ | Full biometric matching, 3D reconstruction, high-confidence recognition |
| **`GOOD`** | $70.0 - 84.9$ | $0.70 - 0.849$ | High-fidelity recognition, attribute extraction, surveillance tracking |
| **`FAIR`** | $50.0 - 69.9$ | $0.50 - 0.699$ | Usable for detection and tracking; recommended threshold for re-capture |
| **`POOR`** | $30.0 - 49.9$ | $0.30 - 0.499$ | Severely degraded; unsuitable for identity verification |
| **`UNUSABLE`** | $< 30.0$ | $< 0.30$ | Deficient (corrupted, pitch-black, extreme blur, microscopic scale) |

---

## 6. Output Schema Structure

Every analyzed face returns both the top-level 0 - 100 score and the explainable factor breakdown:

```json
{
  "quality_score": 88.4,
  "quality": {
    "quality_score": 88.4,
    "overall_quality": 0.884,
    "grade": "EXCELLENT",
    "is_usable": true,
    "factors": {
      "sharpness": 91.2,
      "brightness": 87.5,
      "contrast": 100.0,
      "size": 84.2,
      "landmark": 82.5
    }
  }
}
```
