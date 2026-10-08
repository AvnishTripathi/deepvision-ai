"""
DeepVision AI — Facial Landmark Processing & Geometric Analysis Module
Extracts, validates, and evaluates spatial relationships and symmetry from facial landmarks.
"""

import math
from typing import List, Optional, Tuple, Dict
import numpy as np

from app.schemas.ai import LandmarkPoint
from app.ai.analysis.schemas import FacialLandmarkMetrics


class LandmarkProcessor:
    """
    Processes and validates facial landmarks, extracting geometric metrics
    such as inter-pupillary distance (IPD), eye tilt, and facial symmetry.
    """

    def __init__(self):
        self.expected_landmark_names = {
            "right_eye",
            "left_eye",
            "nose_tip",
            "mouth_right",
            "mouth_left",
        }

    def parse_landmarks_map(self, landmarks: List[LandmarkPoint]) -> Dict[str, Tuple[float, float]]:
        """Maps a list of LandmarkPoints into a dictionary of {name: (x, y)}."""
        result = {}
        for lm in landmarks:
            result[lm.name.lower()] = (float(lm.x), float(lm.y))
        return result

    def compute_metrics(
        self,
        landmarks: List[LandmarkPoint],
    ) -> Optional[FacialLandmarkMetrics]:
        """
        Computes geometric and spatial metrics from 5 canonical facial landmarks:
        - Inter-pupillary distance (IPD)
        - Eye-line tilt angle (roll in degrees)
        - Eye-to-nose distance
        - Nose-to-mouth distance
        - Bilateral facial symmetry ratio
        """
        if not landmarks or len(landmarks) < 5:
            return None

        lm_map = self.parse_landmarks_map(landmarks)

        # Ensure all 5 essential points are present
        if not all(name in lm_map for name in ("right_eye", "left_eye", "nose_tip", "mouth_right", "mouth_left")):
            return None

        re_x, re_y = lm_map["right_eye"]
        le_x, le_y = lm_map["left_eye"]
        n_x, n_y = lm_map["nose_tip"]
        mr_x, mr_y = lm_map["mouth_right"]
        ml_x, ml_y = lm_map["mouth_left"]

        # 1. Inter-pupillary distance (IPD) in pixels
        dx_eyes = le_x - re_x
        dy_eyes = le_y - re_y
        ipd = math.sqrt(dx_eyes * dx_eyes + dy_eyes * dy_eyes)

        # 2. Eye tilt angle (degrees from horizontal)
        # Positive = tilted left side down, negative = tilted right side down
        eye_tilt_rad = math.atan2(dy_eyes, dx_eyes)
        eye_tilt_deg = math.degrees(eye_tilt_rad)

        # 3. Eye to nose distances
        dist_re_n = math.hypot(n_x - re_x, n_y - re_y)
        dist_le_n = math.hypot(n_x - le_x, n_y - le_y)
        mean_eye_to_nose = (dist_re_n + dist_le_n) / 2.0

        # 4. Nose to mouth distances
        dist_n_mr = math.hypot(mr_x - n_x, mr_y - n_y)
        dist_n_ml = math.hypot(ml_x - n_x, ml_y - n_y)
        mean_nose_to_mouth = (dist_n_mr + dist_n_ml) / 2.0

        # 5. Bilateral facial symmetry ratio
        # Compares the distance from nose tip to each eye.
        # Ratio of min/max gives [0.0, 1.0], where 1.0 is perfect bilateral balance.
        max_dist = max(dist_re_n, dist_le_n)
        min_dist = min(dist_re_n, dist_le_n)
        symmetry_ratio = round(min_dist / max_dist, 4) if max_dist > 0.0 else 1.0

        return FacialLandmarkMetrics(
            interpupillary_distance_px=round(ipd, 2),
            eye_tilt_angle_deg=round(eye_tilt_deg, 2),
            eye_to_nose_distance_px=round(mean_eye_to_nose, 2),
            nose_to_mouth_distance_px=round(mean_nose_to_mouth, 2),
            symmetry_ratio=symmetry_ratio,
        )

    def extract_crop_relative_landmarks(
        self,
        landmarks: List[LandmarkPoint],
        crop_origin: Tuple[float, float],
    ) -> List[LandmarkPoint]:
        """
        Translates full-image landmarks to coordinates relative to a cropped face ROI.
        crop_origin is (crop_x1, crop_y1).
        """
        x_off, y_off = crop_origin
        relative_landmarks = []
        for lm in landmarks:
            relative_landmarks.append(
                LandmarkPoint(
                    name=lm.name,
                    x=round(max(0.0, lm.x - x_off), 2),
                    y=round(max(0.0, lm.y - y_off), 2),
                    confidence=lm.confidence,
                )
            )
        return relative_landmarks


# Module-level convenience functions
_default_processor = LandmarkProcessor()


def compute_landmark_metrics(landmarks: List[LandmarkPoint]) -> Optional[FacialLandmarkMetrics]:
    """Computes geometric distance, tilt, and symmetry metrics from 5 facial landmarks."""
    return _default_processor.compute_metrics(landmarks)
