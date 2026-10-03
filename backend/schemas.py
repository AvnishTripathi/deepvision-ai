from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


# ─── Auth ────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    plan: str
    created_at: datetime
    is_active: bool
    api_calls_today: int
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None


# ─── Detection ───────────────────────────────────────────────────────────────

class FaceResult(BaseModel):
    """Result for a single detected face."""
    face_index: int
    bbox: List[int]                        # [x1, y1, x2, y2]
    deepfake_score: float                  # 0.0 (real) → 1.0 (deepfake)
    label: str                             # REAL | SUSPICIOUS | DEEPFAKE
    confidence: float
    signals: Optional[Dict[str, float]] = None   # per-signal scores
    heatmap_base64: Optional[str] = None


class DetectionResult(BaseModel):
    """Full detection result for an image or video."""
    job_id: str
    status: str                            # done | failed | no_faces_found
    overall_score: float
    label: str                             # REAL | SUSPICIOUS | DEEPFAKE
    face_count: Optional[int] = 0
    faces: List[Any] = []                  # list of FaceResult dicts
    heatmap_base64: Optional[str] = None   # annotated image as base64 PNG
    signal_chart_base64: Optional[str] = None   # bar chart as base64 PNG
    processing_time_ms: int = 0
    model_version: str = "v1.0"

    model_config = ConfigDict(protected_namespaces=())  # silence model_ warning


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None        # 0-100 for video jobs
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class VideoDetectResponse(BaseModel):
    """Returned immediately when video is queued."""
    job_id: str
    status: str = "pending"
    message: str = "Video queued for processing"


class UrlDetectRequest(BaseModel):
    url: str



# ─── History ─────────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id: str
    type: str
    status: str
    original_filename: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    overall_score: Optional[float] = None
    label: Optional[str] = None
    processing_time_ms: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)

class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    page: int
    limit: int


# ─── Stats ───────────────────────────────────────────────────────────────────

class UserStats(BaseModel):
    total_scans: int
    deepfakes_found: int
    real_found: int
    suspicious_found: int
    avg_score: float
    api_calls_today: int

