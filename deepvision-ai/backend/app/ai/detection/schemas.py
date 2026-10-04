"""
DeepVision AI — Detection Schemas
Standardized schemas for face bounding boxes, facial landmarks, detection results,
frame inference, and quality assessments.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field

from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    FaceQualityScore,
    FaceDetectionResult,
    FrameInferenceResult,
    PerformanceMetrics,
)

__all__ = [
    "BoundingBox",
    "LandmarkPoint",
    "FaceQualityScore",
    "FaceDetectionResult",
    "FrameInferenceResult",
    "PerformanceMetrics",
]

