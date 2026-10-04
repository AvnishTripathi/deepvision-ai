"""
DeepVision AI — AI Visualization Subpackage
Provides development and testing utilities for rendering face bounding boxes,
confidence badges, and facial landmarks onto image frames.
"""

from app.ai.visualization.annotator import (
    FaceVisualizer,
    annotate_image,
    draw_detections,
)

__all__ = [
    "FaceVisualizer",
    "annotate_image",
    "draw_detections",
]
