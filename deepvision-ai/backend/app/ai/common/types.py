"""
DeepVision AI — Common AI Types & Type Aliases
Provides system-wide type definitions, enumerations, and protocols.
"""

from typing import Union, Tuple, List, Dict, Any, Literal, Protocol, Optional
from enum import Enum
import numpy as np
from PIL import Image

# Image input representations accepted across pipelines
ImageSource = Union[str, bytes, bytearray, np.ndarray, Image.Image]

# Coordinate representations: (x_min, y_min, x_max, y_max)
BoundingBoxCoords = Tuple[float, float, float, float]
Point2D = Tuple[float, float]

# Hardware targets
DeviceTarget = Literal["auto", "cpu", "cuda", "mps", "openvino"]

# Supported model family identifiers
class ModelArchitecture(str, Enum):
    SCRFD = "SCRFD"
    YUNET = "YuNet"
    RETINAFACE = "RetinaFace"
    BASELINE = "Baseline"
    CUSTOM = "Custom"

# Landmark naming identifiers for 5-point facial alignment
FACIAL_LANDMARK_NAMES = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "mouth_left",
    "mouth_right",
)

