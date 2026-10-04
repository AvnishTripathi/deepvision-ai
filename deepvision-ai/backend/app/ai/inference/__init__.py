"""
DeepVision AI — Inference Architecture Package
Provides hardware device resolution, execution provider optimization,
and model runtime engines.
"""

from app.ai.common.types import DeviceType
from app.ai.inference.runtime import InferenceRuntime
from app.ai.inference.device import (
    resolve_device,
    is_gpu_available,
    is_cuda_available,
    get_available_gpu_providers,
    get_optimal_providers,
    get_recommended_thread_count,
    get_device_telemetry,
)
from app.ai.inference.base import BaseInferenceEngine
from app.ai.inference.onnx_engine import ONNXInferenceEngine

__all__ = [
    "DeviceType",
    "InferenceRuntime",
    "resolve_device",
    "is_gpu_available",
    "is_cuda_available",
    "get_available_gpu_providers",
    "get_optimal_providers",
    "get_recommended_thread_count",
    "get_device_telemetry",
    "BaseInferenceEngine",
    "ONNXInferenceEngine",
]
