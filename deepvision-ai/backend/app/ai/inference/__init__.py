"""
DeepVision AI — Inference Architecture Package
Provides hardware device resolution, execution provider optimization,
and model runtime engines.
"""

from app.ai.inference.runtime import InferenceRuntime
from app.ai.inference.device import (
    resolve_device,
    is_cuda_available,
    get_optimal_providers,
    get_recommended_thread_count,
    get_device_telemetry,
)
from app.ai.inference.base import BaseInferenceEngine
from app.ai.inference.onnx_engine import ONNXInferenceEngine

__all__ = [
    "InferenceRuntime",
    "resolve_device",
    "is_cuda_available",
    "get_optimal_providers",
    "get_recommended_thread_count",
    "get_device_telemetry",
    "BaseInferenceEngine",
    "ONNXInferenceEngine",
]

