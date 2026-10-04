"""
DeepVision AI — Inference Engine Abstraction
Abstract Base Class defining standardized lifecycle methods for model runtimes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class BaseInferenceEngine(ABC):
    """Abstract interface for all model execution runtimes."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._is_loaded = False

    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Loads and initializes model weights in memory."""
        pass

    @abstractmethod
    def predict(self, input_tensor: np.ndarray) -> Any:
        """Executes forward-pass inference on the input tensor."""
        pass

    @abstractmethod
    def warmup(self, sample_shape: Tuple[int, ...]) -> None:
        """Executes initial dummy inference to compile kernels and eliminate cold-start latency."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Indicates whether model runtime is loaded and ready for prediction."""
        return self._is_loaded

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns engine and model metadata (inputs, outputs, provider, device)."""
        pass

