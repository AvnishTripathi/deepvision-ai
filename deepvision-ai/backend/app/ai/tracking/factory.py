"""
DeepVision AI — Multi-Face Tracker Factory (Step 5.4)
Instantiates configured tracker instances while keeping tracking algorithm details decoupled.
"""

from typing import Optional, Dict, Any
from app.core.config import settings
from app.ai.tracking.base import BaseFaceTracker
from app.ai.tracking.tracker import ByteTrackFaceTracker


def get_face_tracker(
    tracker_name: str = "bytetrack",
    high_detection_threshold: Optional[float] = None,
    low_detection_threshold: Optional[float] = None,
    match_threshold_first: float = 0.70,
    match_threshold_second: float = 0.50,
    max_time_lost: int = 30,
    min_hits: int = 3,
) -> BaseFaceTracker:
    """
    Factory creating concrete face tracker instances.
    
    Args:
        tracker_name: Identifier of tracker algorithm (default: 'bytetrack').
        high_detection_threshold: Primary detection cutoff (defaults to AI_CONFIDENCE_THRESHOLD).
        low_detection_threshold: Secondary detection floor (defaults to 0.15).
        match_threshold_first: Maximum IoU distance for primary matching (0.70).
        match_threshold_second: Maximum IoU distance for secondary matching (0.50).
        max_time_lost: Consecutive missed frames before tracklet deletion (30).
        min_hits: Consecutive hits to promote tentative track to confirmed (3).
        
    Returns:
        BaseFaceTracker: Initialized multi-face tracking instance.
    """
    name = tracker_name.lower().strip()
    
    high_thresh = (
        high_detection_threshold
        if high_detection_threshold is not None
        else settings.AI_CONFIDENCE_THRESHOLD
    )
    low_thresh = low_detection_threshold if low_detection_threshold is not None else 0.15

    if name in ("bytetrack", "byte_track", "default"):
        return ByteTrackFaceTracker(
            high_detection_threshold=high_thresh,
            low_detection_threshold=low_thresh,
            match_threshold_first=match_threshold_first,
            match_threshold_second=match_threshold_second,
            max_time_lost=max_time_lost,
            min_hits=min_hits,
        )

    raise ValueError(f"Unsupported tracker architecture: '{tracker_name}'. Supported: ['bytetrack'].")
