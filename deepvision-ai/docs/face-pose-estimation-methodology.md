# DeepVision AI — Face Pose & Orientation Estimation Methodology

## 1. Overview & Scope

The **Head Pose & Orientation Estimation** module in DeepVision AI estimates three-dimensional facial orientation angles—**Yaw**, **Pitch**, and **Roll**—directly from 2D facial keypoint landmarks using Perspective-n-Point ($solvePnP$) geometric vision.

> [!CAUTION]
> **Important Engineering & Biometric Disclaimer**:  
> Head pose estimates computed by this system are **engineering approximations designed solely for orientation triage, face alignment preprocessing, and dataset curation**.  
> **Under no circumstances should these estimates be construed as having medical, surgical, clinical, ophthalmological, or certified sub-degree biometric accuracy.**  
> Real-world monocular 2D-to-3D projection is inherently ill-posed without calibrated camera intrinsics, individual craniofacial tomography, or depth sensors.

---

## 2. Pose Output Specification (Step 12)

Orientation is reported as actual calculated continuous floating-point degree values:

```json
{
  "pose": {
    "yaw": 4.2,
    "pitch": -2.1,
    "roll": 1.4
  }
}
```

### Coordinate & Sign Conventions

All angles are expressed in degrees ($^\circ$):

| Angle | Rotation Axis | Positive Sign ($+$) | Negative Sign ($-$) | Typical Range |
| :--- | :---: | :--- | :--- | :---: |
| **Yaw** | Vertical ($Y$-axis) | Turned to the **Subject's Left** (Observer's Right) | Turned to the **Subject's Right** (Observer's Left) | $[-90.0^\circ, +90.0^\circ]$ |
| **Pitch** | Transverse ($X$-axis) | Tilted **Upward** (looking up) | Tilted **Downward** (looking down) | $[-90.0^\circ, +90.0^\circ]$ |
| **Roll** | Optical ($Z$-axis) | Tilted **Clockwise** (tilt right ear to shoulder) | Tilted **Counter-Clockwise** (tilt left ear to shoulder) | $[-90.0^\circ, +90.0^\circ]$ |

---

## 3. Mathematical & Algorithmic Formulation

### 3.1. Canonical 3D Anthropometric Model
To relate 2D pixel landmark detections to a 3D coordinate space, DeepVision AI employs a standardized anthropometric 3D facial reference model derived from anthropometric craniofacial population surveys:

$$\mathbf{P}_{\text{3D}} = \begin{bmatrix}
\mathbf{p}_{\text{right\_eye}} \\
\mathbf{p}_{\text{left\_eye}} \\
\mathbf{p}_{\text{nose\_tip}} \\
\mathbf{p}_{\text{mouth\_right}} \\
\mathbf{p}_{\text{mouth\_left}}
\end{bmatrix} = \begin{bmatrix}
-33.0 & -33.0 & -25.0 \\
+33.0 & -33.0 & -25.0 \\
0.0 & 0.0 & 0.0 \\
-25.0 & +35.0 & -20.0 \\
+25.0 & +35.0 & -20.0
\end{bmatrix}\text{ mm}$$

*Coordinate frame: Origin at the nose tip $\mathbf{p}_{\text{nose\_tip}} = (0, 0, 0)$; $X$ points towards the subject's left eye; $Y$ points downwards towards the mouth; $Z$ points forward.*

### 3.2. Pinhole Camera Intrinsics Approximation
When camera focal length and optical centers are uncalibrated (the standard scenario for arbitrary uploaded imagery), DeepVision AI constructs an idealized pinhole camera projection matrix $\mathbf{K}$:

$$\mathbf{K} = \begin{bmatrix}
f & 0 & c_x \\
0 & f & c_y \\
0 & 0 & 1
\end{bmatrix}$$

Where:
- Focal length: $f = \max(W_{\text{img}}, H_{\text{img}})$
- Principal optical center: $(c_x, c_y) = \left(\frac{W_{\text{img}}}{2}, \frac{H_{\text{img}}}{2}\right)$
- Distortion coefficients vector: $\mathbf{D} = [0, 0, 0, 0]^T$ (zero lens distortion assumption).

### 3.3. Perspective-n-Point ($solvePnP$)
The geometric mapping from 3D canonical points $\mathbf{P}_i$ to observed 2D pixel coordinates $\mathbf{u}_i = (x_i, y_i)^T$ satisfies:

$$s \begin{bmatrix} \mathbf{u}_i \\ 1 \end{bmatrix} = \mathbf{K} \left( \mathbf{R} \, \mathbf{P}_i + \mathbf{t} \right)$$

where $\mathbf{R} \in SO(3)$ is the $3 \times 3$ rotation matrix and $\mathbf{t} \in \mathbb{R}^3$ is the translation vector.

We compute $\mathbf{r}_{\text{vec}}$ and $\mathbf{t}$ via OpenCV's `solvePnP` using the Efficient PnP (`SOLVEPNP_EPNP`) algorithm, falling back to Levenberg-Marquardt optimization (`SOLVEPNP_ITERATIVE`).

### 3.4. Euler Angle Extraction
Using the Rodrigues transform $\mathbf{R} = \text{Rodrigues}(\mathbf{r}_{\text{vec}})$, rotation matrix $\mathbf{R} = [R_{ij}]_{3 \times 3}$ is decomposed into Tait-Bryan Euler angles ($Z$-$Y$-$X$ yaw-pitch-roll convention):

$$\text{Pitch} = \operatorname{atan2}(R_{32}, R_{33})$$
$$\text{Yaw} = \operatorname{atan2}(-R_{31}, \sqrt{R_{11}^2 + R_{21}^2})$$
$$\text{Roll} = \operatorname{atan2}(R_{21}, R_{11})$$

In the presence of gimbal lock singularity ($\sqrt{R_{11}^2 + R_{21}^2} < 10^{-6}$):
$$\text{Pitch} = \operatorname{atan2}(-R_{23}, R_{22}), \quad \text{Yaw} = \operatorname{atan2}(-R_{31}, 0), \quad \text{Roll} = 0$$

All angles in radians are converted to degrees and clamped to physiological limits $[-90.0^\circ, +90.0^\circ]$.

### 3.5. Frontality Score & Pose Categorization
To enable downstream filtering for recognition engines, a continuous frontality score $F \in [0.0, 1.0]$ is derived:

$$F = \max\left(0, 1 - \left( \frac{|\text{Yaw}|}{45.0} \times 0.50 + \frac{|\text{Pitch}|}{35.0} \times 0.30 + \frac{|\text{Roll}|}{30.0} \times 0.20 \right)\right)$$

Discrete categorization:
- **`FRONTAL`**: $|\text{Yaw}| \le 15.0^\circ \land |\text{Pitch}| \le 18.0^\circ$
- **`SLIGHT_TURN`**: $15.0^\circ < |\text{Yaw}| \le 32.0^\circ$
- **`PROFILE`**: $|\text{Yaw}| > 32.0^\circ$
- **`UPWARD`**: $\text{Pitch} > 18.0^\circ$
- **`DOWNWARD`**: $\text{Pitch} < -18.0^\circ$

---

## 4. Documented Unavailable State Contract

> [!IMPORTANT]
> **No Fake Zero Values**:  
> In accordance with Step 12, when reliable head pose estimation cannot be computed, the system **never invents fake zeros** (`yaw=0.0, pitch=0.0, roll=0.0`) or synthetic orientations.  
> Instead, it returns an explicit `null` (`None`) pose and logs a structured failure status code via `get_pose_estimation_status`.

### Failure Reason Taxonomy

| Reason Code | Condition | Behavior |
| :--- | :--- | :--- |
| `insufficient_landmarks` | Detection returned $< 5$ landmarks | `pose: null`, reason logged |
| `degenerate_canvas` | Image dimensions $W \le 0$ or $H \le 0$ | `pose: null`, reason logged |
| `missing_keypoints` | Required 5 points (`right_eye`, `left_eye`, `nose_tip`, `mouth_right`, `mouth_left`) not all found | `pose: null`, reason logged |
| `degenerate_geometry` | Inter-ocular eye distance $< 3.0\text{ px}$ or coordinates contain `NaN`/`Inf` | `pose: null`, reason logged |
| `solve_pnp_unconverged` | PnP optimization failed to converge and geometric triangulation failed | `pose: null`, reason logged |

### JSON Serialization Example

When landmarks are incomplete or occluded:
```json
{
  "pose": null,
  "status": "unavailable",
  "reason": "insufficient_landmarks (requires at least 5 facial landmarks)"
}
```

When landmarks are intact:
```json
{
  "pose": {
    "yaw": 4.2,
    "pitch": -2.1,
    "roll": 1.4
  },
  "frontality_score": 0.941,
  "pose_category": "FRONTAL"
}
```

---

## 5. Verification & Testing

The pose estimation module is verified through automated test suites in `backend/tests/test_face_analysis.py`:
- **`test_head_pose_estimation_angles_dict`**: Confirms `to_angles_dict()` returns exact `{ "yaw": float, "pitch": float, "roll": float }`.
- **`test_head_pose_status_contract`**: Verifies valid inputs yield `(pose, None)` while degraded inputs yield documented `(None, reason)` without raising unhandled exceptions or outputting fake zeros.
- **`test_degenerate_geometry_guard`**: Confirms degenerate coordinates (identical eye points or `NaN` values) return `None`.
