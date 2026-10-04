"""
DeepVision AI — AI & Computer Vision Result Schemas
Strongly typed Pydantic v2 schemas for bounding boxes, landmarks, face detections,
frame inference responses, and performance telemetry.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, computed_field, model_validator, ConfigDict


class NormalizedBoundingBox(BaseModel):
    """Normalized bounding box coordinates bounded between 0.0 and 1.0."""
    x1: float = Field(..., ge=0.0, le=1.0, description="Normalized left coordinate (0.0 to 1.0)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Normalized top coordinate (0.0 to 1.0)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Normalized right coordinate (0.0 to 1.0)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom coordinate (0.0 to 1.0)")

    @computed_field
    @property
    def width(self) -> float:
        return max(0.0, round(self.x2 - self.x1, 4))

    @computed_field
    @property
    def height(self) -> float:
        return max(0.0, round(self.y2 - self.y1, 4))

    def to_pixel(self, image_width: int, image_height: int) -> "BoundingBox":
        """Converts normalized coordinates to absolute image pixel coordinates."""
        return BoundingBox(
            x1=round(self.x1 * image_width, 2),
            y1=round(self.y1 * image_height, 2),
            x2=round(self.x2 * image_width, 2),
            y2=round(self.y2 * image_height, 2),
        )


class BoundingBox(BaseModel):
    """
    Absolute face bounding box coordinates in image space.
    """
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def map_aliases_and_tolerance(cls, data: Any) -> Any:
        """Tolerates legacy x_min, y_min, x_max, y_max input keys."""
        if isinstance(data, dict):
            if "x1" not in data and "x_min" in data:
                data["x1"] = data["x_min"]
            if "y1" not in data and "y_min" in data:
                data["y1"] = data["y_min"]
            if "x2" not in data and "x_max" in data:
                data["x2"] = data["x_max"]
            if "y2" not in data and "y_max" in data:
                data["y2"] = data["y_max"]
        return data

    @computed_field
    @property
    def width(self) -> float:
        return max(0.0, round(self.x2 - self.x1, 4))

    @computed_field
    @property
    def height(self) -> float:
        return max(0.0, round(self.y2 - self.y1, 4))

    @computed_field
    @property
    def area(self) -> float:
        return round(self.width * self.height, 4)

    # Coordinate alias accessors
    @property
    def x_min(self) -> float:
        return self.x1

    @property
    def y_min(self) -> float:
        return self.y1

    @property
    def x_max(self) -> float:
        return self.x2

    @property
    def y_max(self) -> float:
        return self.y2

    def to_normalized(self, image_width: int, image_height: int) -> NormalizedBoundingBox:
        """Converts pixel coordinates to normalized [0.0, 1.0] coordinates."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive non-zero integers.")
        return NormalizedBoundingBox(
            x1=round(max(0.0, min(1.0, self.x1 / image_width)), 4),
            y1=round(max(0.0, min(1.0, self.y1 / image_height)), 4),
            x2=round(max(0.0, min(1.0, self.x2 / image_width)), 4),
            y2=round(max(0.0, min(1.0, self.y2 / image_height)), 4),
        )


class LandmarkPoint(BaseModel):
    """Facial landmark coordinate (e.g. eye, nose tip, mouth corner)."""
    name: str = Field(..., description="Identifier name of the landmark")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Landmark confidence score")


class FaceQualityScore(BaseModel):
    """Multi-metric facial quality assessment."""
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="Composite face quality score (0.0 to 1.0)")
    sharpness: float = Field(..., ge=0.0, description="Laplacian variance sharpness measurement")
    brightness: float = Field(..., ge=0.0, le=255.0, description="Mean luminance brightness (0-255)")
    contrast: float = Field(..., ge=0.0, description="Standard deviation luminance contrast")
    is_usable: bool = Field(True, description="Whether face quality meets minimum processing threshold")


class FaceDetection(BaseModel):
    """
    Standardized face detection result.
    Matches schema:
    {
      "face_id": null,
      "bbox": {"x1": 100, "y1": 80, "x2": 250, "y2": 260},
      "confidence": 0.97,
      "landmarks": null
    }
    """
    face_id: Optional[Union[str, int]] = Field(default=None, description="Optional tracking or identity identifier")
    bbox: BoundingBox = Field(..., description="Face bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    landmarks: Optional[List[LandmarkPoint]] = Field(default=None, description="List of facial landmark points or null")
    normalized_bbox: Optional[NormalizedBoundingBox] = Field(default=None, description="Optional normalized bounding box")
    quality: Optional[FaceQualityScore] = Field(default=None, description="Quality diagnostics")
    track_id: Optional[int] = Field(default=None, description="Tracking identity ID across frame sequences")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom detection metadata")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def map_face_index(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "face_id" not in data and "face_index" in data:
                data["face_id"] = data["face_index"]
        return data

    @property
    def face_index(self) -> Optional[Union[str, int]]:
        return self.face_id


# Alias for backwards compatibility
FaceDetectionResult = FaceDetection


class DetectionResponse(BaseModel):
    """
    Standardized top-level detection response schema.
    """
    image_width: int = Field(..., description="Input image pixel width")
    image_height: int = Field(..., description="Input image pixel height")
    processing_time_ms: float = Field(..., description="Total processing latency in milliseconds")
    device: str = Field(..., description="Device used for inference (e.g. 'cpu', 'gpu')")
    model_name: str = Field(..., description="Name of the detection model used")
    detections: List[FaceDetection] = Field(default_factory=list, description="List of detected faces")

    model_config = ConfigDict(populate_by_name=True)

    @computed_field
    @property
    def face_count(self) -> int:
        return len(self.detections)

    @property
    def faces(self) -> List[FaceDetection]:
        """Convenience property for backwards compatibility."""
        return self.detections


class FrameInferenceResult(DetectionResponse):
    """
    Extended frame inference result containing granular telemetry and pipeline metrics.
    """
    frame_id: Optional[str] = Field(default=None, description="Unique frame identifier or sequence index")
    preprocessing_time_ms: float = Field(default=0.0, description="Preprocessing & letterboxing duration (ms)")
    inference_time_ms: float = Field(default=0.0, description="Model forward-pass latency (ms)")
    postprocessing_time_ms: float = Field(default=0.0, description="Non-maximum suppression & formatting latency (ms)")
    total_time_ms: float = Field(default=0.0, description="End-to-end processing latency (ms)")
    device_used: Optional[str] = Field(default=None, description="Device used alias")
    runtime_provider: str = Field(default="onnxruntime", description="Underlying runtime provider")

    @model_validator(mode="before")
    @classmethod
    def sync_processing_metrics(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync processing_time_ms from total_time_ms if not supplied
            if "processing_time_ms" not in data and "total_time_ms" in data:
                data["processing_time_ms"] = data["total_time_ms"]
            elif "total_time_ms" not in data and "processing_time_ms" in data:
                data["total_time_ms"] = data["processing_time_ms"]

            # Sync device and device_used
            if "device" not in data and "device_used" in data:
                data["device"] = data["device_used"]
            elif "device_used" not in data and "device" in data:
                data["device_used"] = data["device"]

            # Sync faces to detections
            if "detections" not in data and "faces" in data:
                data["detections"] = data["faces"]
            elif "faces" not in data and "detections" in data:
                data["faces"] = data["detections"]
        return data


class PerformanceMetrics(BaseModel):
    """Aggregated inference performance monitoring metrics."""
    frames_processed: int = Field(0, description="Total frames analyzed")
    average_latency_ms: float = Field(0.0, description="Mean inference latency in ms")
    fps: float = Field(0.0, description="Processed frames per second")
    device: str = Field("cpu", description="Active computing hardware")
    runtime: str = Field("CPUExecutionProvider", description="Execution provider used")


class FaceBoundingBox(BaseModel):
    """Face bounding box coordinates in absolute image pixels."""
    x1: float = Field(..., description="Left pixel coordinate")
    y1: float = Field(..., description="Top pixel coordinate")
    x2: float = Field(..., description="Right pixel coordinate")
    y2: float = Field(..., description="Bottom pixel coordinate")

    model_config = ConfigDict(populate_by_name=True)


class DetectedFaceItem(BaseModel):
    """Individual face detection record in development detection response."""
    bbox: FaceBoundingBox = Field(..., description="Face bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    landmarks: Optional[List[LandmarkPoint]] = Field(default=None, description="Optional 5-point facial landmarks")

    model_config = ConfigDict(populate_by_name=True)


class ImageDimensions(BaseModel):
    """Pixel dimensions of the analyzed image."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class AIDetectionResponse(BaseModel):
    """
    Standard development AI detection endpoint response.
    Conforms to:
    {
      "success": true,
      "model": "selected-model",
      "device": "CPU",
      "image": {"width": 1920, "height": 1080},
      "processing_time_ms": 42.5,
      "faces": [{"bbox": {"x1": 100, "y1": 120, "x2": 300, "y2": 350}, "confidence": 0.96}]
    }
    """
    success: bool = Field(True, description="Detection execution status")
    model: str = Field(..., description="Active detection model name")
    device: str = Field(..., description="Inference device used (e.g. CPU, GPU)")
    image: ImageDimensions = Field(..., description="Dimensions of the analyzed image")
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    faces: List[DetectedFaceItem] = Field(default_factory=list, description="List of detected faces")

    model_config = ConfigDict(populate_by_name=True)


class AIDetectionErrorResponse(BaseModel):
    """Development error response returned when detector is not configured or operational."""
    success: bool = Field(False, description="Failure indicator")
    error: str = Field(..., description="Error identifier code")
    message: str = Field(..., description="Human-readable explanation")
    detail: str = Field(..., description="Technical detail message")
    model: Optional[str] = Field(None, description="Target model name")
    device: Optional[str] = Field(None, description="Configured device")
