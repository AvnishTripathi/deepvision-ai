"""
DeepVision AI — Base Face Detector Interface
Defines the abstract interface for face detection architectures, lifecycle management,
and structured inference results.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from app.ai.common.types import ImageSource
from app.ai.detection.schemas import FrameInferenceResult


class BaseFaceDetector(ABC):
    """
    Abstract interface for all face detection engine implementations.
    Enforces the lifecycle contract: load_model -> detect -> unload,
    and guarantees strongly-typed structured results (FrameInferenceResult).
    """

    @abstractmethod
    def load_model(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Loads the detector model weights, graph compilation, and runtime sessions into memory.
        
        Args:
            model_path: Path to ONNX or serialized model weights file.
            device: Target hardware accelerator ('auto', 'cpu', 'gpu').
        """
        pass

    @abstractmethod
    def detect(
        self,
        image_input: ImageSource,
        confidence_threshold: float = 0.5,
        **kwargs: Any,
    ) -> FrameInferenceResult:
        """
        Executes end-to-end face detection on an image input.
        
        Args:
            image_input: Filepath, raw byte stream, NumPy RGB/BGR array, or PIL Image.
            confidence_threshold: Minimum detection confidence score threshold (0.0 to 1.0).
            **kwargs: Implementation-specific detection hyperparameters.
            
        Returns:
            FrameInferenceResult: Standardized, strongly-typed structured result containing
            detected face bounding boxes, landmarks, quality scores, and latency metrics.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unloads the model graph and releases runtime session, CPU/GPU memory, and threads.
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Indicates whether the detector model is currently loaded in memory."""
        pass
