"""
DeepVision AI — Facial Landmark Processing & Geometric Analysis Module
Extracts, validates, normalizes, and evaluates spatial relationships and symmetry from facial landmarks.
Supports structured dictionary format:
{
  "left_eye": {"x": 150.0, "y": 180.0},
  "right_eye": {"x": 220.0, "y": 180.0},
  "nose": {"x": 185.0, "y": 220.0},
  "left_mouth": {"x": 160.0, "y": 260.0},
  "right_mouth": {"x": 210.0, "y": 260.0}
}
Preserves all additional landmarks provided by models without inventing missing points.
All coordinates strictly correspond to the original image coordinate system.
"""

import math
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np

from app.schemas.ai import LandmarkPoint, LandmarkCoordinate
from app.ai.analysis.schemas import FacialLandmarkMetrics


# Standard canonical landmark names (Step 4 specification)
STANDARD_LANDMARK_NAMES = [
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth",
    "right_mouth",
]

# Supported bidirectional aliases mapping alternate conventions to canonical names
LANDMARK_NAME_ALIASES: Dict[str, str] = {
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "nose": "nose",
    "nose_tip": "nose",
    "left_mouth": "left_mouth",
    "mouth_left": "left_mouth",
    "right_mouth": "right_mouth",
    "mouth_right": "right_mouth",
}


def normalize_landmark_name(name: str) -> str:
    """
    Normalizes a landmark identifier name to standard canonical naming:
    'nose_tip' -> 'nose'
    'mouth_left' -> 'left_mouth'
    'mouth_right' -> 'right_mouth'
    Any other landmark name (e.g. 68-point or mesh points) is preserved without modification.
    """
    clean = str(name).strip().lower()
    return LANDMARK_NAME_ALIASES.get(clean, clean)


def landmarks_to_dict(
    landmarks: Union[List[Union[LandmarkPoint, Dict[str, Any]]], Dict[str, Any]],
) -> Dict[str, LandmarkCoordinate]:
    """
    Converts a list or dict of landmarks to a clean structured dictionary formatted as:
    {
      "left_eye": {"x": 150.0, "y": 180.0},
      "right_eye": {"x": 220.0, "y": 180.0},
      "nose": {"x": 185.0, "y": 220.0},
      "left_mouth": {"x": 160.0, "y": 260.0},
      "right_mouth": {"x": 210.0, "y": 260.0}
    }
    Coordinates strictly correspond to the original image coordinate system.
    If the selected model provides more landmarks, all are preserved in the structured dictionary.
    Does not invent unobserved landmarks.
    """
    if not landmarks:
        return {}

    result: Dict[str, LandmarkCoordinate] = {}

    if isinstance(landmarks, dict):
        for raw_name, coord in landmarks.items():
            norm_name = normalize_landmark_name(raw_name)
            result[norm_name] = LandmarkCoordinate.from_point(coord)

    elif isinstance(landmarks, list):
        for lm in landmarks:
            if isinstance(lm, LandmarkPoint):
                norm_name = normalize_landmark_name(lm.name)
                result[norm_name] = LandmarkCoordinate(
                    x=round(float(lm.x), 2),
                    y=round(float(lm.y), 2),
                    confidence=lm.confidence,
                )
            elif isinstance(lm, dict) and "x" in lm and "y" in lm:
                raw_name = lm.get("name", "point")
                norm_name = normalize_landmark_name(raw_name)
                result[norm_name] = LandmarkCoordinate(
                    x=round(float(lm["x"]), 2),
                    y=round(float(lm["y"]), 2),
                    confidence=lm.get("confidence"),
                )

    return result


# Alias for explicit clarity
format_structured_landmarks = landmarks_to_dict


def landmarks_to_list(
    landmarks: Union[Dict[str, Any], List[Union[LandmarkPoint, Dict[str, Any]]]],
) -> List[LandmarkPoint]:
    """Converts structured landmarks into a list of LandmarkPoint items."""
    if not landmarks:
        return []

    if isinstance(landmarks, list):
        return [
            lm if isinstance(lm, LandmarkPoint) else LandmarkPoint(
                name=normalize_landmark_name(lm.get("name", "point")),
                x=float(lm["x"]),
                y=float(lm["y"]),
                confidence=lm.get("confidence"),
            )
            for lm in landmarks
        ]

    result = []
    for name, coord in landmarks.items():
        if isinstance(coord, LandmarkCoordinate):
            result.append(
                LandmarkPoint(
                    name=name,
                    x=coord.x,
                    y=coord.y,
                    confidence=coord.confidence,
                )
            )
        elif isinstance(coord, dict):
            result.append(
                LandmarkPoint(
                    name=name,
                    x=float(coord["x"]),
                    y=float(coord["y"]),
                    confidence=coord.get("confidence"),
                )
            )
        elif isinstance(coord, (tuple, list)) and len(coord) >= 2:
            result.append(LandmarkPoint(name=name, x=float(coord[0]), y=float(coord[1])))
    return result


def validate_and_clip_landmarks(
    landmarks: Union[List[Union[LandmarkPoint, Dict[str, Any]]], Dict[str, Any]],
    image_width: int,
    image_height: int,
    min_confidence: Optional[float] = None,
) -> Dict[str, LandmarkCoordinate]:
    """
    Validates and clips facial landmark coordinates to image canvas boundaries:
    - Enforces 0 <= x <= image_width and 0 <= y <= image_height.
    - Discards landmarks with non-finite coordinates (NaN or Inf).
    - Filters out low-confidence landmarks if min_confidence is specified.
    - Gracefully handles missing landmarks (preserves whatever is valid).
    - Never crashes on missing or partial landmarks.
    """
    if not landmarks or image_width <= 0 or image_height <= 0:
        return {}

    raw_dict = landmarks_to_dict(landmarks)
    validated: Dict[str, LandmarkCoordinate] = {}

    for name, coord in raw_dict.items():
        # Check for NaN / Inf
        if math.isnan(coord.x) or math.isnan(coord.y) or math.isinf(coord.x) or math.isinf(coord.y):
            continue

        # Filter by confidence if threshold is specified
        if min_confidence is not None and coord.confidence is not None:
            if coord.confidence < min_confidence:
                continue

        # Strict clipping to image canvas boundaries [0, image_width], [0, image_height]
        clipped_x = max(0.0, min(float(image_width), float(coord.x)))
        clipped_y = max(0.0, min(float(image_height), float(coord.y)))

        validated[name] = LandmarkCoordinate(
            x=round(clipped_x, 2),
            y=round(clipped_y, 2),
            confidence=coord.confidence,
        )

    return validated


def validate_and_clip_landmarks_list(
    landmarks: Union[List[Union[LandmarkPoint, Dict[str, Any]]], Dict[str, Any]],
    image_width: int,
    image_height: int,
    min_confidence: Optional[float] = None,
) -> List[LandmarkPoint]:
    """
    List representation of validated and boundary-clipped landmark points.
    """
    validated_dict = validate_and_clip_landmarks(
        landmarks=landmarks,
        image_width=image_width,
        image_height=image_height,
        min_confidence=min_confidence,
    )
    return landmarks_to_list(validated_dict)


class LandmarkProcessor:
    """
    Processes and validates facial landmarks, extracting geometric metrics
    such as inter-pupillary distance (IPD), eye tilt, and facial symmetry.
    """

    def __init__(self):
        self.expected_landmark_names = {
            "right_eye",
            "left_eye",
            "nose",
            "nose_tip",
            "left_mouth",
            "right_mouth",
            "mouth_left",
            "mouth_right",
        }

    def parse_landmarks_map(
        self,
        landmarks: Union[List[Union[LandmarkPoint, Dict[str, Any]]], Dict[str, Any]],
    ) -> Dict[str, Tuple[float, float]]:
        """Maps a list or dict of landmarks into a normalized dictionary of {name: (x, y)}."""
        result: Dict[str, Tuple[float, float]] = {}

        if isinstance(landmarks, dict):
            for raw_name, coord in landmarks.items():
                name_clean = str(raw_name).strip().lower()
                if isinstance(coord, (tuple, list)) and len(coord) >= 2:
                    result[name_clean] = (float(coord[0]), float(coord[1]))
                elif isinstance(coord, dict) and "x" in coord and "y" in coord:
                    result[name_clean] = (float(coord["x"]), float(coord["y"]))
                elif hasattr(coord, "x") and hasattr(coord, "y"):
                    result[name_clean] = (float(coord.x), float(coord.y))

        elif isinstance(landmarks, list):
            for lm in landmarks:
                if isinstance(lm, LandmarkPoint):
                    result[lm.name.lower()] = (float(lm.x), float(lm.y))
                elif isinstance(lm, dict) and "name" in lm and "x" in lm and "y" in lm:
                    result[str(lm["name"]).lower()] = (float(lm["x"]), float(lm["y"]))

        # Populate bidirectional aliases
        if "nose" in result and "nose_tip" not in result:
            result["nose_tip"] = result["nose"]
        elif "nose_tip" in result and "nose" not in result:
            result["nose"] = result["nose_tip"]

        if "left_mouth" in result and "mouth_left" not in result:
            result["mouth_left"] = result["left_mouth"]
        elif "mouth_left" in result and "left_mouth" not in result:
            result["left_mouth"] = result["mouth_left"]

        if "right_mouth" in result and "mouth_right" not in result:
            result["mouth_right"] = result["right_mouth"]
        elif "mouth_right" in result and "right_mouth" not in result:
            result["right_mouth"] = result["mouth_right"]

        return result

    def compute_metrics(
        self,
        landmarks: Union[List[Union[LandmarkPoint, Dict[str, Any]]], Dict[str, Any]],
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

        # Ensure all 5 essential facial regions are present (resolves aliases)
        has_eyes = "right_eye" in lm_map and "left_eye" in lm_map
        has_nose = "nose" in lm_map or "nose_tip" in lm_map
        has_mouth = ("left_mouth" in lm_map or "mouth_left" in lm_map) and (
            "right_mouth" in lm_map or "mouth_right" in lm_map
        )

        if not (has_eyes and has_nose and has_mouth):
            return None

        try:
            re_x, re_y = lm_map["right_eye"]
            le_x, le_y = lm_map["left_eye"]
            n_x, n_y = lm_map["nose_tip"]
            mr_x, mr_y = lm_map["mouth_right"]
            ml_x, ml_y = lm_map["mouth_left"]

            # Guard against NaN / Inf values
            coords = [re_x, re_y, le_x, le_y, n_x, n_y, mr_x, mr_y, ml_x, ml_y]
            if any(math.isnan(c) or math.isinf(c) for c in coords):
                return None

            # 1. Inter-pupillary distance (IPD) in pixels
            dx_eyes = le_x - re_x
            dy_eyes = le_y - re_y
            ipd = math.hypot(dx_eyes, dy_eyes)

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
        except Exception:
            return None

    def extract_crop_relative_landmarks(
        self,
        landmarks: Union[List[LandmarkPoint], Dict[str, Any]],
        crop_origin: Tuple[float, float],
    ) -> List[LandmarkPoint]:
        """
        Translates full-image landmarks to coordinates relative to a cropped face ROI.
        crop_origin is (crop_x1, crop_y1).
        """
        x_off, y_off = crop_origin
        lm_list = landmarks_to_list(landmarks)
        relative_landmarks = []
        for lm in lm_list:
            if math.isnan(lm.x) or math.isnan(lm.y) or math.isinf(lm.x) or math.isinf(lm.y):
                continue
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


def compute_landmark_metrics(
    landmarks: Union[List[Union[LandmarkPoint, Dict[str, Any]]], Dict[str, Any]],
) -> Optional[FacialLandmarkMetrics]:
    """Computes geometric distance, tilt, and symmetry metrics from facial landmarks."""
    return _default_processor.compute_metrics(landmarks)
