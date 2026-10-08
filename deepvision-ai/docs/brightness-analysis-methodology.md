# Face Brightness & Illumination Analysis Methodology

## 1. Overview & Objective

The DeepVision AI Face Analysis layer evaluates the illumination, exposure quality, and contrast distribution of detected face regions. For every detected face crop, the system calculates:

- **`brightness_score`**: Continuous normalized exposure suitability score ($0.0 \dots 1.0$, where $1.0$ represents optimal facial illumination and $0.0$ represents unusable underexposure or sensor clipping).
- **`brightness_status`**: Qualitative discrete classification:
  - `too_dark`
  - `acceptable`
  - `too_bright`
- **`brightness`**: Mean luminance across the face crop in native 8-bit scale ($0.0 \dots 255.0$).
- **`contrast`**: Standard deviation of pixel intensities across the face crop.

---

## 2. Mathematical Formulation & Luminance Calculation

### 2.1 Color Conversion (ITU-R BT.601 Standard)
The RGB cropped face region $I_{\text{RGB}}(x, y)$ is converted to single-channel 8-bit grayscale luminance $Y(x, y)$ using the ITU-R Recommendation BT.601 standard weights, which model human photopic spectral sensitivity (peaking in green, lower in red and blue):
$$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

### 2.2 Mean Brightness
The mean brightness represents the first central moment (average intensity) across all $N$ pixels in the face crop:
$$\text{brightness} = \mu_Y = \frac{1}{N} \sum_{x=1}^{W} \sum_{y=1}^{H} Y(x, y)$$

### 2.3 Contrast (Standard Deviation)
Facial contrast is measured as the sample standard deviation of luminance, reflecting the dynamic range between facial highlights (bridge of nose, forehead) and shadows (eye sockets, jawline):
$$\text{contrast} = \sigma_Y = \sqrt{\frac{1}{N} \sum_{x=1}^{W} \sum_{y=1}^{H} (Y(x, y) - \mu_Y)^2}$$

---

## 3. Threshold Assumptions & Physical Justification

The thresholds separating `too_dark`, `acceptable`, and `too_bright` are rooted in digital sensor dynamic range and biometric face recognition standards (e.g. ISO/IEC 19794-5 face image quality specifications):

| Status (`brightness_status`) | Default Threshold | Configuration Parameter | Physical Rationale & Assumption |
|---|---|---|---|
| **`too_dark`** | $\text{brightness} < 60.0$ | `AI_BRIGHTNESS_MIN_ACCEPTABLE` | **Shadow Clipping**: At $\mu_Y < 60.0$ ($< 23.5\%$ dynamic range), camera sensors encounter the noise floor and quantize facial details into crushed blacks. Irises, lip contours, and nostril boundaries lose gradient contrast. |
| **`acceptable`** | $60.0 \le \text{brightness} \le 200.0$ | (Between min and max thresholds) | **Balanced Exposure**: Normal human skin tones across diverse Fitzpatrick types exhibit average luminance between 60.0 and 200.0 under diffuse natural or office lighting without clipping highlights. |
| **`too_bright`** | $\text{brightness} > 200.0$ | `AI_BRIGHTNESS_MAX_ACCEPTABLE` | **Specular Highlight Clipping**: At $\mu_Y > 200.0$ ($> 78.4\%$ dynamic range), forehead, nose bridge, and cheek highlights saturate at $255$. Digital blowout irreversibly erases high-frequency biometric landmark contours. |

---

## 4. Score Formulation (`brightness_score`)

The `brightness_score` provides a continuous $[0.0, 1.0]$ metric suitable for downstream ranking, quality filtering, and composite scoring:

1. **Underexposed Domain** ($\text{brightness} < \text{min\_acceptable}$):
   $$\text{brightness\_score} = \max\left(0.0, \frac{\text{brightness}}{\text{min\_acceptable}} \times 0.70\right)$$
   Scales linearly from $0.0$ (pitch black) up to $0.70$ at the threshold boundary.

2. **Acceptable Exposure Domain** ($\text{min\_acceptable} \le \text{brightness} \le \text{max\_acceptable}$):
   Centered at the midpoint $\mu_{\text{opt}} = \frac{\text{min} + \text{max}}{2} = 130.0$, scaling smoothly between $0.70$ and $1.0$:
   $$\text{half\_range} = \frac{\text{max} - \text{min}}{2} = 70.0$$
   $$\text{deviation} = \frac{|\text{brightness} - \mu_{\text{opt}}|}{\text{half\_range}}$$
   $$\text{brightness\_score} = \max(0.70, 1.0 - 0.30 \times \text{deviation})$$

3. **Overexposed Domain** ($\text{brightness} > \text{max\_acceptable}$):
   $$\text{denom} = 255.0 - \text{max\_acceptable} = 55.0$$
   $$\text{brightness\_score} = \max\left(0.0, \frac{255.0 - \text{brightness}}{\text{denom}} \times 0.70\right)$$
   Scales linearly down from $0.70$ at the boundary to $0.0$ at complete white-out ($255.0$).

---

## 5. Configuration & Overrides

Thresholds can be adjusted dynamically at three levels without code modifications:

1. **Environment Variables**:
   ```env
   AI_BRIGHTNESS_MIN_ACCEPTABLE=60.0
   AI_BRIGHTNESS_MAX_ACCEPTABLE=200.0
   ```
2. **Global Application Settings**:
   Configured in `app.core.config.settings.AI_BRIGHTNESS_MIN_ACCEPTABLE` and `AI_BRIGHTNESS_MAX_ACCEPTABLE`.
3. **Function-Level Execution Parameters**:
   Directly overridable in Python calls:
   ```python
   from app.ai.analysis.metrics import calculate_face_brightness

   metrics = calculate_face_brightness(
       face_crop_rgb=crop,
       min_acceptable=50.0,   # Stricter or more lenient shadow threshold
       max_acceptable=215.0,  # Custom highlight threshold
   )
   print(metrics.brightness_status)  # 'too_dark', 'acceptable', or 'too_bright'
   print(metrics.brightness_score)   # 0.0 to 1.0
   ```
