"""
DeepVision AI — Hardware Device & Execution Provider Management
Determines optimal runtime hardware (AUTO, CPU, GPU) and manages execution provider fallbacks.
Designed to never crash if GPU acceleration is unavailable.
"""

import os
import multiprocessing
import logging
from typing import List, Dict, Any, Union, Optional
from app.ai.common.types import DeviceType, DeviceTarget

logger = logging.getLogger("deepvision.ai.device")

# Known GPU execution providers across diverse hardware vendors (NVIDIA, AMD, Intel, Apple)
GPU_EXECUTION_PROVIDERS = (
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",       # DirectML on Windows (supports NVIDIA, AMD, Intel Arc)
    "ROCMExecutionProvider",      # AMD ROCm
    "CoreMLExecutionProvider",    # Apple Silicon
    "OpenVINOExecutionProvider",  # Intel OpenVINO / GPU
)


def get_available_gpu_providers() -> List[str]:
    """
    Scans active ONNX Runtime installation for supported GPU execution providers.
    Does NOT assume or hard-code CUDA.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        return [p for p in GPU_EXECUTION_PROVIDERS if p in available]
    except (ImportError, Exception) as exc:
        logger.debug("Failed querying ONNX Runtime providers: %s", exc)
        return []


def is_cuda_available() -> bool:
    """Checks whether NVIDIA CUDA acceleration is active and accessible."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except (ImportError, Exception):
        pass

    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return True
    except (ImportError, Exception):
        pass

    return False


def is_gpu_available() -> bool:
    """
    Checks whether ANY hardware GPU acceleration runtime is available.
    Supports CUDA, DirectML, ROCm, CoreML, and PyTorch backends.
    """
    # 1. Check ONNX Runtime GPU providers
    gpu_providers = get_available_gpu_providers()
    if gpu_providers:
        return True

    # 2. Check PyTorch CUDA or MPS backends
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except (ImportError, Exception):
        pass

    return False


def resolve_device(requested_device: Union[DeviceType, str] = DeviceType.AUTO) -> str:
    """
    Resolves 'auto', 'cpu', or 'gpu' against real system hardware.

    Behavior:
    - AUTO: Automatically selects best supported hardware. If GPU is available, selects GPU;
            otherwise gracefully falls back to CPU.
    - CPU: Explicitly selects CPU.
    - GPU: Selects GPU if supported. If GPU is unavailable, logs a notice and gracefully
           falls back to CPU. Never raises an unhandled crash or terminates.
    """
    # Normalize input
    if isinstance(requested_device, DeviceType):
        target = requested_device.value
    else:
        target = str(requested_device).lower().strip()

    # Normalize aliases (e.g. cuda -> gpu)
    if target in ("auto", "default"):
        if is_gpu_available():
            detected = get_available_gpu_providers()
            logger.info("Auto-detection selected GPU acceleration (providers: %s).", detected)
            return "gpu"
        else:
            logger.info("Auto-detection: No GPU hardware detected. Gracefully selecting optimized CPU runtime.")
            return "cpu"

    elif target in ("gpu", "cuda", "dml", "rocm", "mps"):
        if is_gpu_available():
            logger.info("GPU acceleration requested and confirmed available.")
            return "gpu"
        else:
            logger.warning(
                "GPU acceleration requested ('%s'), but no compatible GPU execution provider "
                "or driver was detected on this host. Gracefully falling back to CPU runtime.",
                target,
            )
            return "cpu"

    elif target in ("cpu",):
        logger.debug("CPU execution explicitly selected.")
        return "cpu"

    # Fallback for unexpected string inputs
    logger.warning("Unrecognized device target '%s'. Falling back safely to CPU.", target)
    return "cpu"


def get_optimal_providers(device: str = "cpu") -> List[str]:
    """
    Returns prioritized list of ONNX Runtime execution providers for target device.
    Ensures CPUExecutionProvider is ALWAYS included as the final safety net.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        available = ["CPUExecutionProvider"]

    device_norm = device.lower().strip()

    if device_norm == "gpu":
        active_gpu_providers = [p for p in GPU_EXECUTION_PROVIDERS if p in available]
        if active_gpu_providers:
            # Prioritized GPU providers followed by CPU fallback
            return active_gpu_providers + ["CPUExecutionProvider"]

    elif device_norm == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    elif device_norm == "openvino" and "OpenVINOExecutionProvider" in available:
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

    # Default to universal CPU
    return ["CPUExecutionProvider"]


def get_recommended_thread_count() -> int:
    """
    Calculates optimal thread pool allocation for CPU inference.
    Prevents CPU thrashing on multi-core host architectures.
    """
    cores = multiprocessing.cpu_count()
    return max(1, min(4, cores))


def get_device_telemetry() -> Dict[str, Any]:
    """
    Returns comprehensive runtime telemetry regarding hardware detection,
    active device resolution, and available execution providers.
    """
    resolved = resolve_device(DeviceType.AUTO)
    gpu_providers = get_available_gpu_providers()
    
    return {
        "active_device": resolved,
        "gpu_available": is_gpu_available(),
        "cuda_available": is_cuda_available(),
        "available_gpu_providers": gpu_providers,
        "cpu_cores": multiprocessing.cpu_count(),
        "recommended_threads": get_recommended_thread_count(),
        "optimal_providers": get_optimal_providers(resolved),
    }
