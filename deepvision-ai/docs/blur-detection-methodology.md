# Face Blur Detection & Clarity Estimation Methodology

## 1. Overview & Objective

The DeepVision AI Face Analysis layer evaluates the focus, sharpness, and motion/defocus blur of detected face regions. For every detected face crop, the system calculates:

- **`blur_score`**: Normalized continuous focus/clarity score ($0.0 \dots 1.0$, where $1.0$ represents crisp focus and $0.0$ represents heavy blur).
- **`blur_status`**: Qualitative discrete classification:
  - `sharp`
  - `slightly_blurred`
  - `blurry`
- **`sharpness`**: Raw variance of the Laplacian filter response.

---

## 2. Mathematical Formulation & Algorithm

The blur detection engine utilizes the **Variance of the Laplacian** method (Pech-Pacheco et al.), a classical, high-efficiency spatial domain operator that requires zero GPU acceleration and executes in $< 1\text{ms}$ on CPU.

### 2.1 Color Conversion
The input cropped face region $I_{\text{RGB}}$ is converted to single-channel 8-bit grayscale luminance $I(x, y)$:
$$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

### 2.2 Laplacian Convolving
The discrete 2D Laplacian operator computes the second spatial derivative of the luminance intensity image:
$$L(x, y) = \nabla^2 I(x, y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

Convolved using the standard $3 \times 3$ isotropic second-derivative kernel:
$$K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

### 2.3 Variance Measurement
The raw sharpness metric is calculated as the statistical variance of the Laplacian response across all $N$ pixels in the face crop:
$$\mu_L = \frac{1}{N} \sum_{x, y} L(x, y)$$
$$\text{sharpness} = \text{Var}(L) = \frac{1}{N} \sum_{x, y} \left( L(x, y) - \mu_L \right)^2$$

High variance indicates a high proportion of high-frequency edges and rapid gradient transitions (crisp eyelashes, distinct iris boundaries, sharp lips). Low variance indicates smooth, attenuated gradients characteristic of defocus or motion blur.

### 2.4 Score Normalization
The raw variance is normalized to an explainable continuous probability range $[0.0, 1.0]$:
$$\text{blur\_score} = \min\left(1.0, \max\left(0.0, \frac{\text{sharpness} - 15.0}{185.0}\right)\right)$$

---

## 3. Configurable Threshold Assumptions

Thresholds separating sharp, slightly blurred, and blurry faces are configurable both globally via application settings and per-call via function arguments:

| Status (`blur_status`) | Default Threshold | Configuration Key | Description |
|---|---|---|---|
| **`sharp`** | $\text{sharpness} \ge 100.0$ | `AI_BLUR_SHARP_THRESHOLD` | Crisp high-frequency features (pores, lashes, iris texture). Suitable for high-accuracy recognition. |
| **`slightly_blurred`** | $50.0 \le \text{sharpness} < 100.0$ | `AI_BLUR_SLIGHTLY_BLURRED_THRESHOLD` | Moderate sharpness; minor motion or defocus blur present. Generally usable for detection and pose tracking. |
| **`blurry`** | $\text{sharpness} < 50.0$ | (Below slightly blurred threshold) | Significant motion or defocus blur; high-frequency facial edge detail is suppressed. |

### Configuration Sources:
1. **Environment Variables**:
   ```env
   AI_BLUR_SHARP_THRESHOLD=100.0
   AI_BLUR_SLIGHTLY_BLURRED_THRESHOLD=50.0
   ```
2. **Settings**: Configured in `app.core.config.settings.AI_BLUR_SHARP_THRESHOLD` and `AI_BLUR_SLIGHTLY_BLURRED_THRESHOLD`.
3. **Function Arguments**: Overridable dynamically in `estimate_face_blur(crop, sharp_threshold=..., slightly_blurred_threshold=...)`.

---

## 4. Critical Technical Caveats: Why Blur Score is NOT a Universal Quality Metric

> [!WARNING]
> **Important Engineering Principle**:
> The Laplacian variance blur score measures **high-frequency spatial variation**, NOT universal or subjective face quality. It must never be treated as a comprehensive standalone quality metric.

### Key Factors Limiting Laplacian Variance Interpretation:
1. **Skin Smoothness & Cosmetics**: Naturally smooth youth skin, heavy foundation, cosmetic powder, or beautification filters diminish high-frequency texture, yielding lower variance even when the subject is perfectly in focus.
2. **Lighting Conditions**: Diffuse ambient studio lighting produces softer gradient transitions than direct harsh lighting, reducing Laplacian variance without degrading image utility.
3. **Sensor Noise & Grain**: High ISO sensor noise, compression artifacts, and checkerboard frequency patterns artificially inflate Laplacian variance, producing deceptively high "sharpness" scores on noisy, low-quality sensors.
4. **Motion Blur vs. Defocus Blur**: The Laplacian operator is isotropic and does not distinguish directional motion blur from radial optical defocus blur.
5. **Resolution Dependency**: Very small faces ($< 32\text{px}$) have fewer pixels to exhibit high-frequency gradients, naturally reducing variance compared to close-up portrait crops ($> 300\text{px}$).

For these reasons, DeepVision AI combines the blur measurement into a multi-dimensional composite quality evaluation along with **face size**, **illumination & contrast**, **3D head pose**, and **landmark symmetry**.
