"""
DeepVision AI — 3D Head Pose & Orientation Estimation Module
Estimates 3D head pose angles (Yaw, Pitch, Roll) and frontality score
from 2D facial landmarks using Perspective-n-Point (solvePnP).
"""

import math
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

from app.schemas.ai import LandmarkPoint
from app.ai.analysis.schemas import HeadPoseEstimation, PoseCategory


# 3D Anthropometric Model Points (in millimeters, camera frame: X-right, Y-down, Z-front)
# Canonical human facial geometry referenced from anthropometric survey standards:
CANONICAL_3D_MODEL_POINTS = np.array(
    [
        [-33.0, -33.0, -25.0],  # Right Eye (X-left on face, Y-above nose)
        [33.0, -33.0, -25.0],   # Left Eye (X-right on face, Y-above nose)
        [0.0, 0.0, 0.0],        # Nose Tip (Origin)
        [-25.0, 35.0, -20.0],   # Mouth Right Corner (Y-below nose)
        [25.0, 35.0, -20.0],    # Mouth Left Corner (Y-below nose)
    ],
    dtype=np.float64,
)


class HeadPoseEstimator:
    """
    Estimates 3D head pose orientation (Yaw, Pitch, Roll) from 5-point 2D facial landmarks.
    Solves the Perspective-n-Point (PnP) problem against a canonical 3D facial model.
    """

    def __init__(self, model_points: Optional[np.ndarray] = None):
        self.model_points = model_points if model_points is not None else CANONICAL_3D_MODEL_POINTS

    def estimate_pose(
        self,
        landmarks: List[LandmarkPoint],
        image_shape: Tuple[int, int],
    ) -> Optional[HeadPoseEstimation]:
        """
        Estimates Yaw, Pitch, Roll Euler angles and frontality score.

        Args:
            landmarks: 5 facial landmarks (right_eye, left_eye, nose_tip, mouth_right, mouth_left).
            image_shape: (height, width) of the image in pixels.

        Returns:
            HeadPoseEstimation containing yaw, pitch, roll in degrees,
            frontality_score (0.0 to 1.0), and pose_category.
        """
        if not landmarks or len(landmarks) < 5:
            return None

        img_h, img_w = image_shape
        if img_h <= 0 or img_w <= 0:
            return None

        # Build 2D landmark point array mapped in matching order
        lm_map: Dict[str, Tuple[float, float]] = {
            lm.name.lower(): (float(lm.x), float(lm.y)) for lm in landmarks
        }

        required_names = ("right_eye", "left_eye", "nose_tip", "mouth_right", "mouth_left")
        if not all(name in lm_map for name in required_names):
            return None

        image_points = np.array(
            [
                lm_map["right_eye"],
                lm_map["left_eye"],
                lm_map["nose_tip"],
                lm_map["mouth_right"],
                lm_map["mouth_left"],
            ],
            dtype=np.float64,
        )

        # Approximate pinhole camera intrinsic matrix based on image dimensions
        focal_length = float(max(img_w, img_h))
        center_x = float(img_w) / 2.0
        center_y = float(img_h) / 2.0

        camera_matrix = np.array(
            [
                [focal_length, 0.0, center_x],
                [0.0, focal_length, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume zero lens distortion

        # Solve Perspective-n-Point
        try:
            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP if hasattr(cv2, "SOLVEPNP_EPNP") else cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            if not success:
                return self._geometric_heuristic_pose_fallback(lm_map)

            # Convert rotation vector to 3x3 rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Compute Euler angles from rotation matrix
            # Convention: Yaw (Y-axis), Pitch (X-axis), Roll (Z-axis)
            sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
            singular = sy < 1e-6

            if not singular:
                pitch = math.atan2(rmat[2, 1], rmat[2, 2])
                yaw = math.atan2(-rmat[2, 0], sy)
                roll = math.atan2(rmat[1, 0], rmat[0, 0])
            else:
                pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
                yaw = math.atan2(-rmat[2, 0], sy)
                roll = 0.0

            pitch_deg = round(math.degrees(pitch), 2)
            yaw_deg = round(math.degrees(yaw), 2)
            roll_deg = round(math.degrees(roll), 2)

            # Sanity clamp to realistic physiological head rotation range [-90, +90]
            pitch_deg = max(-90.0, min(90.0, pitch_deg))
            yaw_deg = max(-90.0, min(90.0, yaw_deg))
            roll_deg = max(-90.0, min(90.0, roll_deg))

            frontality = self._compute_frontality_score(yaw_deg, pitch_deg, roll_deg)
            category = self._classify_pose(yaw_deg, pitch_deg, roll_deg)

            return HeadPoseEstimation(
                yaw=yaw_deg,
                pitch=pitch_deg,
                roll=roll_deg,
                frontality_score=frontality,
                pose_category=category,
            )

        except Exception:
            # Graceful geometric fallback if OpenCV solvePnP encounters singularity
            return self._geometric_heuristic_pose_fallback(lm_map)

    def _compute_frontality_score(self, yaw: float, pitch: float, roll: float) -> float:
        """
        Computes a normalized frontality score [0.0, 1.0].
        1.0 represents a face directly oriented toward the camera.
        """
        # Weighted penalties: yaw has highest weight for facial recognition / quality
        penalty = (
            (abs(yaw) / 45.0) * 0.50
            + (abs(pitch) / 35.0) * 0.30
            + (abs(roll) / 30.0) * 0.20
        )
        score = max(0.0, min(1.0, 1.0 - penalty))
        return round(score, 3)

    def _classify_pose(self, yaw: float, pitch: float, roll: float) -> PoseCategory:
        """Categorizes head pose based on Euler angle thresholds."""
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        if abs_yaw > 32.0:
            return PoseCategory.PROFILE
        if abs_yaw > 15.0:
            return PoseCategory.SLIGHT_TURN
        if pitch > 18.0:
            return PoseCategory.UPWARD
        if pitch < -18.0:
            return PoseCategory.DOWNWARD
        return PoseCategory.FRONTAL

    def _geometric_heuristic_pose_fallback(
        self,
        lm_map: Dict[str, Tuple[float, float]],
    ) -> HeadPoseEstimation:
        """
        Pure geometric heuristic fallback when solvePnP fails.
        Estimates yaw from nose-to-eye horizontal distance ratio,
        roll from eye tilt, and pitch from nose-to-eye vertical ratio.
        """
        re_x, re_y = lm_map["right_eye"]
        le_x, le_y = lm_map["left_eye"]
        n_x, n_y = lm_map["nose_tip"]

        # Roll: eye-line angle
        roll_deg = math.degrees(math.atan2(le_y - re_y, le_x - re_x))

        # Yaw: asymmetry between left-eye-to-nose and right-eye-to-nose
        d_re = n_x - re_x
        d_le = le_x - n_x
        tot = max(1.0, d_re + d_le)
        asymmetry = (d_re - d_le) / tot  # [-1, 1]
        yaw_deg = asymmetry * 45.0

        # Pitch: vertical offset of nose relative to eye midpoint
        mid_eye_y = (re_y + le_y) / 2.0
        pitch_deg = max(-30.0, min(30.0, (n_y - mid_eye_y - 20.0) * 0.5))

        pitch_deg = round(pitch_deg, 2)
        yaw_deg = round(yaw_deg, 2)
        roll_deg = round(roll_deg, 2)

        frontality = self._compute_frontality_score(yaw_deg, pitch_deg, roll_deg)
        category = self._classify_pose(yaw_deg, pitch_deg, roll_deg)

        return HeadPoseEstimation(
            yaw=yaw_deg,
            pitch=pitch_deg,
            roll=roll_deg,
            frontality_score=frontality,
            pose_category=category,
        )


_default_pose_estimator = HeadPoseEstimator()


def estimate_head_pose(
    landmarks: List[LandmarkPoint],
    image_shape: Tuple[int, int],
) -> Optional[HeadPoseEstimation]:
    """Estimates 3D head pose angles (Yaw, Pitch, Roll) and frontality score."""
    return _default_pose_estimator.estimate_pose(landmarks, image_shape)
