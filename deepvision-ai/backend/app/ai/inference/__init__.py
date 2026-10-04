"""
DeepVision AI — Inference Package Initialization
"""

from app.ai.inference.base import BaseInferenceEngine
from app.ai.inference.onnx_engine import ONNXInferenceEngine

__all__ = [
    "BaseInferenceEngine",
    "ONNXInferenceEngine",
]
