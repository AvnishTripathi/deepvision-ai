"""
DeepVision AI — Application AI Service Layer
Provides clean application-level orchestration between API endpoints and the AI engine.
Maintains strict architectural separation: API -> AI Service -> Detector Interface -> Inference Runtime -> Model.
"""

import logging
import threading
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from app.core.config import settings, Settings
from app.core.exceptions import ImageTooLargeException
from app.ai.common.types import ImageSource
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.detector import FaceDetector
from app.ai.detection.schemas import DetectionResponse, FrameInferenceResult
from app.ai.inference.device import get_device_telemetry
from app.services.face_analysis_service import FaceAnalysisService
from app.schemas.ai import DetectedFaceItem

logger = logging.getLogger("deepvision.service.ai")


class AIService:
    """
    Application-level AI Service.
    Decouples API controllers from underlying model runtime, detector implementations,
    and face analysis diagnostics.
    """

    def __init__(
        self,
        detector: Optional[BaseFaceDetector] = None,
        analysis_service: Optional[FaceAnalysisService] = None,
        config: Optional[Settings] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or settings
        self._detector = detector or FaceDetector(
            model_path=model_path,
            device=self.config.AI_DEVICE,
            target_size=(640, 640),
        )
        self._analysis_service = analysis_service or FaceAnalysisService()
        logger.info(
            "AIService initialized with device: %s, model: %s, model_path: %s",
            self.config.AI_DEVICE,
            self.config.AI_MODEL_NAME,
            model_path,
        )

    @property
    def analysis_service(self) -> FaceAnalysisService:
        """Access the underlying FaceAnalysisService."""
        return self._analysis_service

    @property
    def detector(self) -> BaseFaceDetector:
        """Access the underlying detector implementing BaseFaceDetector interface."""
        return self._detector

    @property
    def is_operational(self) -> bool:
        """
        Indicates whether the underlying detector has active, operational model weights loaded.
        Returns False if no physical model weights are loaded in the inference runtime.
        """
        return bool(getattr(self._detector, "is_loaded", False))

    def detect_faces(
        self,
        image_input: ImageSource,
        confidence_threshold: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> DetectionResponse:
        """
        Application-level face detection entrypoint.
        
        Args:
            image_input: Image bytes, filepath, PIL Image, or NumPy array.
            confidence_threshold: Optional override for minimum detection confidence.
            request_id: Optional trace ID for correlation in structured logs.
            
        Returns:
            DetectionResponse: Standardized, strongly typed detection response.
        """
        # 1. Enforce payload size limits before processing
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) > self.config.ai_max_image_bytes:
                raise ImageTooLargeException(
                    f"Image payload size ({len(image_input)} bytes) exceeds configured limit "
                    f"of {self.config.AI_MAX_IMAGE_SIZE_MB}MB ({self.config.ai_max_image_bytes} bytes)."
                )

        # 2. Resolve confidence threshold against configuration default
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.AI_CONFIDENCE_THRESHOLD
        )

        # 3. Execute through the abstract detector interface
        result: FrameInferenceResult = self._detector.detect(
            image_input=image_input,
            confidence_threshold=threshold,
        )

        logger.debug(
            "Detected %d face(s) in %sms on device %s [request_id=%s]",
            result.face_count,
            result.processing_time_ms,
            result.device,
            request_id or "untracked",
        )

        return result
    def _resolve_image_rgb(self, image_input: ImageSource) -> np.ndarray:
        """Safely normalizes any ImageSource input into an in-memory RGB NumPy array."""
        if isinstance(image_input, np.ndarray):
            return image_input
        preprocessor = getattr(self._detector, "preprocessor", None)
        if preprocessor is not None and hasattr(preprocessor, "load_image"):
            return preprocessor.load_image(image_input)
        from app.ai.detection.preprocessor import ImagePreprocessor
        return ImagePreprocessor().load_image(image_input)

    def detect_and_analyze_faces(
        self,
        image_input: ImageSource,
        confidence_threshold: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[List[DetectedFaceItem], FrameInferenceResult]:
        """
        Executes face detection followed by detailed face analysis via FaceAnalysisService.
        Architecture:
            API -> AI Service -> Face Detector -> Face Analysis Service
        """
        # 1. Enforce payload size limits before processing
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) > self.config.ai_max_image_bytes:
                raise ImageTooLargeException(
                    f"Image payload size ({len(image_input)} bytes) exceeds configured limit "
                    f"of {self.config.AI_MAX_IMAGE_SIZE_MB}MB ({self.config.ai_max_image_bytes} bytes)."
                )

        # 2. Resolve image to RGB numpy array safely
        image_rgb = self._resolve_image_rgb(image_input)
        orig_h, orig_w = image_rgb.shape[:2]

        # 3. Detect faces via detector
        detection_result: FrameInferenceResult = self.detect_faces(
            image_input=image_rgb,
            confidence_threshold=confidence_threshold,
            request_id=request_id,
        )

        canvas_h = detection_result.image_height or orig_h
        canvas_w = detection_result.image_width or orig_w

        det_time_ms = round(
            detection_result.inference_time_ms
            if (detection_result.inference_time_ms is not None and detection_result.inference_time_ms > 0.0)
            else (detection_result.total_time_ms or detection_result.processing_time_ms or 0.0),
            2
        )

        # 4. Analyze detected faces via dedicated FaceAnalysisService
        face_items, timing_breakdown = self._analysis_service.analyze_detected_faces_with_timing(
            image_rgb=image_rgb,
            detections=detection_result.detections,
            image_shape=(canvas_h, canvas_w),
            detection_time_ms=det_time_ms,
        )

        # Attach measured stage metrics (Step 18)
        detection_result.detection_time_ms = timing_breakdown.detection_time_ms
        detection_result.landmark_time_ms = timing_breakdown.landmark_time_ms
        detection_result.quality_analysis_time_ms = timing_breakdown.quality_analysis_time_ms
        detection_result.pose_time_ms = timing_breakdown.pose_time_ms
        detection_result.total_analysis_time_ms = timing_breakdown.total_analysis_time_ms

        return face_items, detection_result

    def analyze_faces(
        self,
        image_input: ImageSource,
        confidence_threshold: Optional[float] = None,
        request_id: Optional[str] = None,
        min_sharpness: float = 50.0,
    ) -> Tuple[Any, FrameInferenceResult]:
        """
        Executes end-to-end face detection followed by structured face analysis
        via FaceAnalysisService.
        """
        # 1. Enforce payload size limits before processing
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) > self.config.ai_max_image_bytes:
                raise ImageTooLargeException(
                    f"Image payload size ({len(image_input)} bytes) exceeds configured limit "
                    f"of {self.config.AI_MAX_IMAGE_SIZE_MB}MB ({self.config.ai_max_image_bytes} bytes)."
                )

        # 2. Resolve image to RGB numpy array safely
        image_rgb = self._resolve_image_rgb(image_input)
        orig_h, orig_w = image_rgb.shape[:2]

        # 3. Detect faces
        detection_result: FrameInferenceResult = self.detect_faces(
            image_input=image_rgb,
            confidence_threshold=confidence_threshold,
            request_id=request_id,
        )

        canvas_h = detection_result.image_height or orig_h
        canvas_w = detection_result.image_width or orig_w

        det_time_ms = round(
            detection_result.inference_time_ms
            if (detection_result.inference_time_ms is not None and detection_result.inference_time_ms > 0.0)
            else (detection_result.total_time_ms or detection_result.processing_time_ms or 0.0),
            2
        )

        # 4. Perform comprehensive structured face analysis via FaceAnalysisService
        analyses, timing_breakdown = self._analysis_service.analyze_structured_detections_with_timing(
            image_rgb=image_rgb,
            detections=detection_result.detections,
            image_shape=(canvas_h, canvas_w),
            detection_time_ms=det_time_ms,
        )

        # Attach measured stage metrics (Step 18)
        detection_result.detection_time_ms = timing_breakdown.detection_time_ms
        detection_result.landmark_time_ms = timing_breakdown.landmark_time_ms
        detection_result.quality_analysis_time_ms = timing_breakdown.quality_analysis_time_ms
        detection_result.pose_time_ms = timing_breakdown.pose_time_ms
        detection_result.total_analysis_time_ms = timing_breakdown.total_analysis_time_ms

        return analyses, detection_result

    def get_service_status(self) -> Dict[str, Any]:
        """Returns runtime telemetry, active device resolution, and model status."""
        telemetry = get_device_telemetry()
        return {
            "service": "ai_inference",
            "model_name": self.config.AI_MODEL_NAME,
            "model_version": self.config.AI_MODEL_VERSION,
            "configured_device": self.config.AI_DEVICE,
            "gpu_enabled": self.config.AI_ENABLE_GPU,
            "is_model_loaded": self._detector.is_loaded,
            "load_count": getattr(self._detector, "load_count", 0),
            "inference_count": getattr(self._detector, "inference_count", 0),
            "telemetry": telemetry,
        }

    def shutdown(self) -> None:
        """Gracefully unloads detector models and releases memory resources."""
        logger.info("Shutting down AIService detector sessions.")
        self._detector.unload()


# Module-level thread-safe singleton instance and mutex
_ai_service_instance: Optional[AIService] = None
_ai_service_lock = threading.Lock()


def init_ai_service() -> AIService:
    """
    Initializes and warms up the AIService singleton once during application startup.
    Ensures model weights are loaded once into memory and reused across all requests.
    Thread-safe.
    """
    global _ai_service_instance
    with _ai_service_lock:
        if _ai_service_instance is None:
            model_path = settings.resolve_model_path()
            _ai_service_instance = AIService(model_path=model_path)
            logger.info(
                "AIService singleton initialized at application startup [model=%s, is_operational=%s, load_count=%s]",
                settings.AI_MODEL_NAME,
                _ai_service_instance.is_operational,
                getattr(_ai_service_instance.detector, "load_count", 1),
            )
    return _ai_service_instance


def get_ai_service() -> AIService:
    """
    Dependency provider / accessor for the AIService singleton.
    Implements double-checked locking: reuses the existing loaded detector without reloading.
    """
    if _ai_service_instance is None:
        return init_ai_service()
    return _ai_service_instance


def shutdown_ai_service() -> None:
    """
    Gracefully shuts down and unloads the AIService singleton during application shutdown.
    Thread-safe.
    """
    global _ai_service_instance
    with _ai_service_lock:
        if _ai_service_instance is not None:
            _ai_service_instance.shutdown()
            _ai_service_instance = None
            logger.info("AIService singleton successfully shut down.")

