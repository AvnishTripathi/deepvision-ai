"""
DeepVision AI — AI & Computer Vision Result Schemas
Strongly typed Pydantic v2 schemas for bounding boxes, landmarks, quality scores,
detection results, frame analytics, and performance telemetry.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, computed_field


class BoundingBox(BaseModel):
    """Normalized or absolute face bounding coordinates."""
    x_min: float = Field(..., description="Top-left X coordinate")
    y_min: float = Field(..., description="Top-left Y coordinate")
    x_max: float = Field(..., description="Bottom-right X coordinate")
    y_max: float = Field(..., description="Bottom-right Y coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")

    @computed_field
    @property
    def width(self) -> float:
        return max(0.0, round(self.x_max - self.x_min, 4))

    @computed_field
    @property
    def height(self) -> float:
        return max(0.0, round(self.y_max - self.y_min, 4))

    @computed_field
    @property
    def area(self) -> float:
        return round(self.width * self.height, 4)


class LandmarkPoint(BaseModel):
    """Facial landmark coordinates (e.g. eye, nose, mouth corners)."""
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


class FaceDetectionResult(BaseModel):
    """Detailed detection result for an individual face in a frame."""
    face_index: int = Field(..., description="Index position of face in the frame")
    bbox: BoundingBox = Field(..., description="Facial bounding box coordinates")
    landmarks: List[LandmarkPoint] = Field(default_factory=list, description="5 or 68 landmark points")
    quality: Optional[FaceQualityScore] = Field(None, description="Quality diagnostics")
    track_id: Optional[int] = Field(None, description="Tracking identity ID across frame sequences")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom detection metadata")


class FrameInferenceResult(BaseModel):
    """Comprehensive analysis result for an image or video frame."""
    frame_id: Optional[str] = Field(None, description="Unique frame identifier or sequence index")
    image_width: int = Field(..., description="Input image pixel width")
    image_height: int = Field(..., description="Input image pixel height")
    face_count: int = Field(0, description="Total detected faces in frame")
    faces: List[FaceDetectionResult] = Field(default_factory=list, description="List of detected faces")
    
    # Timing & Performance
    preprocessing_time_ms: float = Field(0.0, description="Preprocessing & letterboxing duration (ms)")
    inference_time_ms: float = Field(0.0, description="Model forward-pass latency (ms)")
    postprocessing_time_ms: float = Field(0.0, description="Non-maximum suppression & formatting latency (ms)")
    total_time_ms: float = Field(0.0, description="End-to-end processing latency (ms)")
    
    # System Telemetry
    device_used: str = Field("cpu", description="Inference device (e.g. cpu, cuda)")
    model_name: str = Field("deepvision-detector", description="Active detection model name")
    runtime_provider: str = Field("onnxruntime", description="Underlying runtime provider")


class PerformanceMetrics(BaseModel):
    """Aggregated inference performance monitoring metrics."""
    frames_processed: int = Field(0, description="Total frames analyzed")
    average_latency_ms: float = Field(0.0, description="Mean inference latency in ms")
    fps: float = Field(0.0, description="Processed frames per second")
    device: str = Field("cpu", description="Active computing hardware")
    runtime: str = Field("CPUExecutionProvider", description="Execution provider used")

