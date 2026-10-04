"""
DeepVision AI — Detection Package Initialization
"""

from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.detector import FaceDetector

__all__ = [
    "BaseFaceDetector",
    "FaceDetector",
]
