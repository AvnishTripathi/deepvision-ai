# DeepVision AI — Face Quality Classification Methodology

## 1. Executive Summary & Objectives

The **Face Quality Classification** module (Step 13) provides a standardized, categorical assessment that translates the continuous, measurable **Face Quality Score (0 - 100)** into clear operational tiers:

- **`excellent`**
- **`good`**
- **`acceptable`**
- **`poor`**

> [!IMPORTANT]
> **Deterministic & Measurable Foundation**:  
> In DeepVision AI, quality classification is **not** an arbitrary black-box prediction. It is calculated directly from the transparent, multi-factor Face Quality Score established in Step 11 (combining 25% sharpness, 20% photopic brightness, 15% dynamic contrast, 20% face resolution/scale, and 20% landmark geometry with bottleneck defect gating).

---

## 2. Classification Tiers & Operational Guidance

| Classification | Score Interval (0 - 100) | Optical & Geometric Characteristics | Recommended Pipeline Action |
| :--- | :---: | :--- | :--- |
| **`excellent`** | **Score $\ge 80.0$** | Sharp high-frequency facial edges ($\sigma_{\Delta}^2 \ge 100$), balanced photopic luminance ($60 \le \mu_Y \le 200$), ample resolution ($D_{\min} \ge 120\text{ px}$), and symmetric 5-point facial keypoints. | **Biometric Enrollment & 1:N Search**: Unrestricted use for canonical template generation, passport/ID verification, and high-security authentication. |
| **`good`** | **$65.0 \le \text{Score} < 80.0$** | High visual fidelity with minor deviations (e.g. mild lighting variation, slight compression, or minor pose turn). | **Automated 1:1 Verification**: Direct authorization for access control, automated surveillance logging, and face verification. |
| **`acceptable`** | **$45.0 \le \text{Score} < 65.0$** | Noticeable blur, sub-optimal exposure, or modest resolution ($32\text{ px} \le D_{\min} < 80\text{ px}$). Discernible facial structure preserved. | **Secondary Review / Soft Triage**: Usable with caution; recommended for candidate proposal generation or fallback human-in-the-loop review. |
| **`poor`** | **Score $< 45.0$** | Severe optical defects (extreme defocus/motion blur, pitch-black darkness $< 40$, specular blowout $> 215$, sub-32px thumbnail scale, or missing landmarks). | **Rejection & Re-Capture**: Unusable for reliable biometric inference. Flagged for user re-capture or discarded from dataset ingestion. |

---

## 3. Mathematical Mapping & Algorithm

The classification function maps the input quality score $Q \in [0, 100]$ to the discrete tier:

$$\text{Classification}(Q) = \begin{cases}
\textbf{excellent}, & \text{if } Q \ge T_{\text{excellent}} \\
\textbf{good}, & \text{if } T_{\text{good}} \le Q < T_{\text{excellent}} \\
\textbf{acceptable}, & \text{if } T_{\text{acceptable}} \le Q < T_{\text{good}} \\
\textbf{poor}, & \text{if } Q < T_{\text{acceptable}}
\end{cases}$$

### Scale-Agnostic Normalization Support
The classification engine supports both **$0 - 100$ integer/float scales** and **normalized $0.0 - 1.0$ float scales**:

$$Q_{100} = \begin{cases}
Q \times 100.0, & \text{if } 0.0 < Q \le 1.0 \\
Q, & \text{otherwise}
\end{cases}$$

This guarantees seamless compatibility across legacy microservices returning unit intervals and modern endpoints returning percentage scores.

---

## 4. Configurable Threshold Architecture

Thresholds can be adjusted dynamically at runtime via programmatic arguments or statically via application settings and environment variables.

### 4.1. Environment Variables & App Settings

Configured in `app/core/config.py`:

```ini
# Face Quality Classification Thresholds (0.0 to 100.0)
AI_QUALITY_EXCELLENT_THRESHOLD=80.0
AI_QUALITY_GOOD_THRESHOLD=65.0
AI_QUALITY_ACCEPTABLE_THRESHOLD=45.0
```

### 4.2. Programmatic Override

```python
from app.ai.analysis.quality import classify_face_quality
from app.ai.analysis.schemas import QualityClassification

# Strict enterprise enrollment profile:
tier = classify_face_quality(
    quality_score=78.5,
    excellent_threshold=85.0,
    good_threshold=70.0,
    acceptable_threshold=55.0,
)
# Returns: QualityClassification.GOOD
```

---

## 5. Structured Output & API Integration

The quality classification is exposed directly on both [`StructuredFaceAnalysis`](file:///c:/Users/abhis/Music/Deep%20Face%20Detection/deepvision-ai/backend/app/ai/analysis/schemas.py#L422-L470) and [`ComprehensiveFaceQuality`](file:///c:/Users/abhis/Music/Deep%20Face%20Detection/deepvision-ai/backend/app/ai/analysis/schemas.py#L360-L420):

```json
{
  "face_id": "face_0",
  "bbox": {
    "x1": 120,
    "y1": 85,
    "x2": 260,
    "y2": 245
  },
  "confidence": 0.985,
  "quality_score": 84.6,
  "classification": "excellent",
  "visibility": "excellent",
  "quality": {
    "quality_score": 84.6,
    "overall_quality": 0.846,
    "classification": "excellent",
    "grade": "EXCELLENT",
    "sharpness": 142.3,
    "brightness": 128.4,
    "contrast": 54.1,
    "is_usable": true,
    "factors": {
      "sharpness": 88.5,
      "brightness": 94.2,
      "contrast": 100.0,
      "size": 85.0,
      "landmark": 87.5
    }
  }
}
```

### Backward Compatibility
The legacy `grade` field (`EXCELLENT`, `GOOD`, `FAIR`, `POOR`, `UNUSABLE`) remains fully preserved alongside `classification` (`excellent`, `good`, `acceptable`, `poor`), ensuring zero breaking changes for existing consumers.
