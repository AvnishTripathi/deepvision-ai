"""
DeepVision AI — ONNX Runtime Inference Engine
Generic, high-performance ONNX model runner with optimized CPU threading.
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import onnxruntime as ort
from app.ai.inference.base import BaseInferenceEngine

logger = logging.getLogger("deepvision.inference.onnx")


class ONNXInferenceEngine(BaseInferenceEngine):
    """Production ONNX Runtime engine optimized for CPU execution."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "cpu",
        num_threads: int = 4,
    ):
        super().__init__(model_name=model_name, device=device)
        self.model_path = model_path
        self.num_threads = num_threads
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Initializes ONNX InferenceSession with optimized execution providers."""
        target_path = model_path or self.model_path
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(f"ONNX model file not found at: {target_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = self.num_threads

        # Preferred execution providers for Windows Intel CPU
        providers = ["CPUExecutionProvider"]

        logger.info("Initializing ONNX session for %s with providers: %s", self.model_name, providers)
        self.session = ort.InferenceSession(target_path, sess_options=session_options, providers=providers)

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self._is_loaded = True
        logger.info("ONNX engine loaded successfully. Inputs: %s, Outputs: %s", self.input_names, self.output_names)

    def predict(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Runs forward-pass evaluation on input tensor."""
        if not self._is_loaded or self.session is None:
            raise RuntimeError(f"Engine {self.model_name} is not loaded.")

        input_feed = {self.input_names[0]: input_tensor}
        return self.session.run(self.output_names, input_feed)

    def warmup(self, sample_shape: Tuple[int, ...] = (1, 3, 640, 640)) -> None:
        """Warms up the ONNX session with dummy tensor data."""
        if not self._is_loaded or self.session is None:
            return
        dummy = np.zeros(sample_shape, dtype=np.float32)
        try:
            self.predict(dummy)
            logger.info("Warmup pass completed for %s", self.model_name)
        except Exception as exc:
            logger.warning("Warmup failed for %s: %s", self.model_name, exc)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns engine architecture and provider telemetry."""
        return {
            "model_name": self.model_name,
            "engine": "ONNXRuntime",
            "version": ort.__version__,
            "device": self.device,
            "providers": self.session.get_providers() if self.session else [],
            "input_names": self.input_names,
            "output_names": self.output_names,
            "is_loaded": self._is_loaded,
        }
