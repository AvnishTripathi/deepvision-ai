"""
DeepVision AI — Hardware Device & Execution Provider Management
Determines optimal runtime hardware (CPU, CUDA, MPS) and thread pool parameters.
"""

import os
import multiprocessing
import logging
from typing import List, Dict, Any
from app.ai.common.types import DeviceTarget

logger = logging.getLogger("deepvision.ai.device")


def is_cuda_available() -> bool:
    """Checks whether CUDA runtime is accessible through installed frameworks."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        pass

    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        pass

    return False


def resolve_device(requested_device: DeviceTarget = "auto") -> str:
    """
    Resolves 'auto' or explicit device targets against actual system capabilities.
    Falls back gracefully to 'cpu' if requested acceleration is unavailable.
    """
    req = str(requested_device).lower().strip()
    if req in ("auto", "cuda", "gpu"):
        if is_cuda_available():
            logger.info("CUDA acceleration detected and enabled.")
            return "cuda"
        else:
            logger.info("No CUDA hardware detected; defaulting to optimized CPU execution.")
            return "cpu"
    elif req in ("cpu", "mps", "openvino"):
        return req
    return "cpu"


def get_optimal_providers(device: str = "cpu") -> List[str]:
    """
    Returns prioritized list of ONNX Runtime execution providers for target device.
    """
    import onnxruntime as ort
    available = ort.get_available_providers()

    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "openvino" and "OpenVINOExecutionProvider" in available:
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def get_recommended_thread_count() -> int:
    """
    Calculates optimal thread pool allocation for CPU inference.
    Prevents CPU thrashing on mobile/laptop architectures.
    """
    cores = multiprocessing.cpu_count()
    # Allocate cores without choking web request handlers
    return max(1, min(4, cores))


def get_device_telemetry() -> Dict[str, Any]:
    """Returns runtime telemetry regarding hardware detection."""
    resolved = resolve_device("auto")
    return {
        "active_device": resolved,
        "cuda_available": is_cuda_available(),
        "cpu_cores": multiprocessing.cpu_count(),
        "recommended_threads": get_recommended_thread_count(),
        "optimal_providers": get_optimal_providers(resolved),
    }

