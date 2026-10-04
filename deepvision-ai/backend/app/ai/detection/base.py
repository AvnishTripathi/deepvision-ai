"""
DeepVision AI — Base Face Detector Interface
Defines the standard contract for face detection engines across images and video frames.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from PIL import Image
from app.schemas.ai import FrameInferenceResult


class BaseFaceDetector(ABC):
    """Abstract interface for all face detection implementations."""

    @abstractmethod
    def detect(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image],
        confidence_threshold: float = 0.5,
    ) -> FrameInferenceResult:
        """
        Executes end-to-end face detection on an image input.
        
        Args:
            image_input: Filepath, byte buffer, RGB array, or PIL image.
            confidence_threshold: Minimum detection confidence score.
            
        Returns:
            FrameInferenceResult containing detected faces, bounding boxes,
            facial landmarks, quality analytics, and timing latency.
        """
        pass

