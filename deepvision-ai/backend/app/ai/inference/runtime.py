"""
DeepVision AI — Model Inference Runtime
Encapsulates ONNX Runtime execution sessions, session options, warmup passes,
and multi-threaded CPU/GPU forward evaluation.
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import onnxruntime as ort

from app.ai.inference.device import (
    resolve_device,
    get_optimal_providers,
    get_recommended_thread_count,
)

logger = logging.getLogger("deepvision.ai.runtime")


class InferenceRuntime:
    """
    Standardized, robust model execution runtime wrapping ONNX Runtime.
    Decoupled from specific network architectures.
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "auto",
        num_threads: Optional[int] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = resolve_device(device)
        self.num_threads = num_threads or get_recommended_thread_count()
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self._is_loaded = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: Optional[str] = None) -> None:
        """Loads and compiles model graph into an active execution session."""
        target_path = model_path or self.model_path
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = self.num_threads
        session_options.inter_op_num_threads = 1

        providers = get_optimal_providers(self.device)
        logger.info(
            "Loading %s (%s threads, device: %s, providers: %s)",
            self.model_name,
            self.num_threads,
            self.device,
            providers,
        )

        try:
            self.session = ort.InferenceSession(
                target_path,
                sess_options=session_options,
                providers=providers,
            )
        except Exception as exc:
            logger.warning(
                "Execution provider initialization failed with %s: %s. Falling back safely to CPUExecutionProvider.",
                providers,
                exc,
            )
            self.session = ort.InferenceSession(
                target_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self._is_loaded = True
        logger.info("Runtime initialized for %s. Inputs: %s, Outputs: %s", self.model_name, self.input_names, self.output_names)

    def predict(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Runs forward-pass evaluation on input tensor."""
        if not self._is_loaded or self.session is None:
            raise RuntimeError(f"InferenceRuntime for {self.model_name} is not loaded.")

        feed_dict = {self.input_names[0]: input_tensor}
        return self.session.run(self.output_names, feed_dict)

    def warmup(self, sample_shape: Tuple[int, ...] = (1, 3, 640, 640)) -> float:
        """
        Executes dummy forward-pass to initialize execution kernels.
        Returns warmup latency in milliseconds.
        """
        if not self._is_loaded or self.session is None:
            return 0.0

        start = time.perf_counter()
        dummy = np.zeros(sample_shape, dtype=np.float32)
        try:
            self.predict(dummy)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info("Warmup pass completed for %s in %sms", self.model_name, elapsed_ms)
            return elapsed_ms
        except Exception as exc:
            logger.warning("Warmup failed for %s: %s", self.model_name, exc)
            return 0.0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_metadata(self) -> Dict[str, Any]:
        """Provides runtime telemetry and model introspection."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "threads": self.num_threads,
            "providers": self.session.get_providers() if self.session else [],
            "input_names": self.input_names,
            "output_names": self.output_names,
            "is_loaded": self._is_loaded,
        }

