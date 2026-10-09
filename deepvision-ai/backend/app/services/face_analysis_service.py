"""
DeepVision AI — Dedicated Face Analysis Service (Step 15)
Provides an enterprise service layer decoupling facial analysis from API routers.
Architecture:
    API -> AI Service -> Face Detector -> Face Analysis Service
                                            ├── Landmarks
                                            ├── Blur
                                            ├── Brightness
                                            ├── Contrast
                                            ├── Size
                                            ├── Pose
                                            └── Quality Score
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from app.core.config import settings
from app.schemas.ai import (
    BoundingBox,
    FaceBoundingBox,
    LandmarkPoint,
    LandmarkCoordinate,
    DetectedFaceItem,
    FaceDetectionQuality,
    DetectedFacePose,
)
from app.ai.detection.schemas import FaceDetection
from app.ai.analysis.schemas import (
    StructuredFaceAnalysis,
    ComprehensiveFaceQuality,
    FaceSizeMetrics,
    ClarityMetrics,
    IlluminationMetrics,
    FacialLandmarkMetrics,
    HeadPoseEstimation,
    FaceQualityFactors,
    QualityGrade,
    QualityClassification,
    BlurStatus,
    BrightnessStatus,
    FaceVisibility,
    FaceVisibilityAssessment,
)
from app.ai.analysis.landmarks import (
    validate_and_clip_landmarks,
    validate_and_clip_landmarks_list,
    compute_landmark_metrics,
)
from app.ai.analysis.pose import (
    estimate_head_pose,
    get_pose_estimation_status,
)
from app.ai.analysis.metrics import (
    compute_face_size_metrics,
    compute_clarity_metrics,
    compute_illumination_metrics,
    calculate_face_visibility,
)
from app.ai.analysis.quality import (
    calculate_face_quality_score,
    classify_face_quality,
    evaluate_quality_grade,
)

logger = logging.getLogger("deepvision.services.analysis")


class FaceAnalysisService:
    """
    Dedicated Face Analysis Service.
    Orchestrates modular optical and geometric diagnostics on detected faces:
    - Landmarks extraction and normalization
    - Blur detection via Laplacian variance
    - Photopic brightness evaluation
    - Contrast dynamic range assessment
    - Face scale and area ratio computation
    - 3D Head pose orientation (solvePnP)
    - Multi-factor transparent quality scoring and classification
    """

    def __init__(self, min_sharpness: float = 50.0):
        self.min_sharpness = min_sharpness

    # -------------------------------------------------------------------------
    # 1. Modular Sub-Component Analyzers
    # -------------------------------------------------------------------------

    def extract_and_validate_landmarks(
        self,
        landmarks: Optional[Union[List[LandmarkPoint], Dict[str, Any]]],
        image_shape: Tuple[int, int],
    ) -> Tuple[Dict[str, LandmarkCoordinate], List[LandmarkPoint]]:
        """
        Validates, bounds-clips, and formats facial landmarks.
        Returns:
            Tuple of (landmarks_dict, landmarks_list).
        """
        img_h, img_w = image_shape
        if not landmarks:
            return {}, []

        lms_dict = validate_and_clip_landmarks(landmarks, image_width=img_w, image_height=img_h)
        lms_list = validate_and_clip_landmarks_list(landmarks, image_width=img_w, image_height=img_h)
        return lms_dict, lms_list

    def analyze_blur(
        self,
        face_crop: np.ndarray,
        min_sharpness_threshold: Optional[float] = None,
    ) -> Tuple[float, str, ClarityMetrics]:
        """
        Estimates blur using Laplacian variance.
        Returns:
            Tuple of (blur_score, blur_status_str, ClarityMetrics).
        """
        threshold = min_sharpness_threshold if min_sharpness_threshold is not None else self.min_sharpness
        clarity = compute_clarity_metrics(face_crop, min_sharpness_threshold=threshold)
        blur_score = round(clarity.sharpness, 1)
        blur_status = clarity.blur_status.value
        return blur_score, blur_status, clarity

    def analyze_brightness(
        self,
        face_crop: np.ndarray,
    ) -> Tuple[float, str, IlluminationMetrics]:
        """
        Calculates mean photopic luminance.
        Returns:
            Tuple of (brightness_score, brightness_status_str, IlluminationMetrics).
        """
        illum = compute_illumination_metrics(face_crop)
        brightness_score = round(illum.brightness, 1)
        brightness_status = illum.brightness_status.value
        return brightness_score, brightness_status, illum

    def analyze_contrast(
        self,
        illumination_metrics: IlluminationMetrics,
    ) -> Tuple[float, str]:
        """
        Evaluates dynamic range from luminance standard deviation.
        Returns:
            Tuple of (contrast_score, contrast_status_str).
        """
        contrast_score = round(illumination_metrics.contrast, 1)
        if illumination_metrics.contrast < 16.0:
            contrast_status = "low_contrast"
        elif illumination_metrics.contrast > 75.0:
            contrast_status = "high_contrast"
        else:
            contrast_status = "acceptable"
        return contrast_score, contrast_status

    def analyze_size(
        self,
        bbox: BoundingBox,
        image_shape: Tuple[int, int],
        ipd: Optional[float] = None,
    ) -> FaceSizeMetrics:
        """Computes scale, physical pixel dimensions, and canvas area ratio."""
        return compute_face_size_metrics(bbox, image_shape, ipd=ipd)

    def estimate_pose(
        self,
        landmarks_list: List[LandmarkPoint],
        image_shape: Tuple[int, int],
    ) -> Tuple[Optional[Dict[str, float]], Optional[HeadPoseEstimation]]:
        """
        Estimates 3D head pose orientation (Yaw, Pitch, Roll) via solvePnP.
        Returns (angles_dict, HeadPoseEstimation) or (None, None) if unavailable.
        """
        if not landmarks_list or len(landmarks_list) < 5:
            return None, None

        pose_obj = estimate_head_pose(landmarks_list, image_shape)
        if pose_obj is None:
            return None, None

        angles_dict = pose_obj.to_angles_dict()
        return angles_dict, pose_obj

    def calculate_quality(
        self,
        face_crop: np.ndarray,
        bbox: BoundingBox,
        image_shape: Tuple[int, int],
        clarity_metrics: ClarityMetrics,
        illumination_metrics: IlluminationMetrics,
        size_metrics: FaceSizeMetrics,
        landmarks_list: List[LandmarkPoint],
        detection_confidence: float = 0.95,
        landmark_metrics: Optional[FacialLandmarkMetrics] = None,
    ) -> Tuple[float, str, FaceQualityFactors, QualityGrade]:
        """
        Calculates transparent 0 - 100 face quality score and qualitative classification.
        """
        q_score_100, factors, grade = calculate_face_quality_score(
            face_crop_rgb=face_crop,
            bbox=bbox,
            image_shape=image_shape,
            clarity_metrics=clarity_metrics,
            illumination_metrics=illumination_metrics,
            size_metrics=size_metrics,
            landmark_metrics=landmark_metrics,
            landmarks=landmarks_list,
            detection_confidence=detection_confidence,
        )
        classification = classify_face_quality(q_score_100).value
        return q_score_100, classification, factors, grade

    # -------------------------------------------------------------------------
    # 2. End-to-End Face Detection Item Formatter (Step 14 & 15 Contract)
    # -------------------------------------------------------------------------

    def analyze_face_detection_item(
        self,
        image_rgb: np.ndarray,
        detection: FaceDetection,
        image_shape: Tuple[int, int],
    ) -> DetectedFaceItem:
        """
        Processes a single raw face detection and returns a fully analyzed DetectedFaceItem.
        Guarantees isolation: corrupted coordinates or missing landmarks never crash the batch.
        """
        img_h, img_w = image_shape
        bbox = detection.bbox

        # 1. Landmarks
        lms_dict, lms_list = self.extract_and_validate_landmarks(detection.landmarks, image_shape)

        # 2. Extract safe face crop
        x1_int = max(0, int(round(bbox.x1)))
        y1_int = max(0, int(round(bbox.y1)))
        x2_int = min(img_w, int(round(bbox.x2)))
        y2_int = min(img_h, int(round(bbox.y2)))
        face_crop = (
            image_rgb[y1_int:y2_int, x1_int:x2_int]
            if x2_int > x1_int and y2_int > y1_int
            else np.empty((0, 0, 3), dtype=np.uint8)
        )

        # 3. Size metrics
        size_metrics = self.analyze_size(bbox, image_shape)
        face_area_ratio = round(size_metrics.face_area_ratio, 4)

        # 4. Blur & sharpness
        blur_score, blur_status, clarity = self.analyze_blur(face_crop)

        # 5. Brightness & exposure
        brightness_score, brightness_status, illum = self.analyze_brightness(face_crop)

        # 6. Contrast & dynamic range
        contrast_score, contrast_status = self.analyze_contrast(illum)

        # 7. Quality scoring & classification
        q_score_100, q_classification, factors, grade = self.calculate_quality(
            face_crop=face_crop,
            bbox=bbox,
            image_shape=image_shape,
            clarity_metrics=clarity,
            illumination_metrics=illum,
            size_metrics=size_metrics,
            landmarks_list=lms_list,
            detection_confidence=detection.confidence,
        )

        # 8. Visibility assessment
        vis_assessment = calculate_face_visibility(
            face_crop_rgb=face_crop,
            bbox=bbox,
            image_shape=image_shape,
            size_metrics=size_metrics,
            clarity_metrics=clarity,
            illumination_metrics=illum,
            landmarks=lms_list,
            detection_confidence=detection.confidence,
        )
        visibility_val = vis_assessment.visibility.value

        # Build FaceDetectionQuality
        quality_obj = FaceDetectionQuality(
            score=round(q_score_100, 1),
            classification=q_classification,
            blur_score=blur_score,
            blur_status=blur_status,
            brightness_score=brightness_score,
            brightness_status=brightness_status,
            contrast_score=contrast_score,
            contrast_status=contrast_status,
            face_area_ratio=face_area_ratio,
            visibility=visibility_val,
            overall_quality=round(q_score_100 / 100.0, 3),
            sharpness=blur_score,
            brightness=brightness_score,
            contrast=contrast_score,
            is_usable=q_score_100 >= 30.0,
        )

        # 9. 3D Head pose
        pose_dict, _ = self.estimate_pose(lms_list, image_shape)

        return DetectedFaceItem(
            face_id=detection.face_id,
            bbox=FaceBoundingBox(
                x1=round(bbox.x1, 2),
                y1=round(bbox.y1, 2),
                x2=round(bbox.x2, 2),
                y2=round(bbox.y2, 2),
            ),
            confidence=round(detection.confidence, 4),
            landmarks=lms_dict if lms_dict else None,
            landmarks_list=lms_list if lms_list else None,
            landmarks_dict=lms_dict if lms_dict else None,
            quality=quality_obj,
            pose=pose_dict,
        )

    def analyze_detected_faces(
        self,
        image_rgb: np.ndarray,
        detections: List[FaceDetection],
        image_shape: Tuple[int, int],
    ) -> List[DetectedFaceItem]:
        """
        Analyzes a batch of detected faces across an image.
        Applies fault isolation so an anomaly in one face never crashes the batch.
        """
        results: List[DetectedFaceItem] = []
        for d in detections:
            try:
                item = self.analyze_face_detection_item(image_rgb, d, image_shape)
                results.append(item)
            except Exception as exc:
                logger.error("Error analyzing face detection %s: %s", getattr(d, "face_id", "unknown"), exc, exc_info=True)
                # Robust fallback item ensuring caller receives valid record
                try:
                    fallback_item = DetectedFaceItem(
                        face_id=getattr(d, "face_id", None),
                        bbox=FaceBoundingBox(
                            x1=round(d.bbox.x1, 2),
                            y1=round(d.bbox.y1, 2),
                            x2=round(d.bbox.x2, 2),
                            y2=round(d.bbox.y2, 2),
                        ),
                        confidence=round(float(getattr(d, "confidence", 0.0)), 4),
                        landmarks=None,
                        landmarks_list=None,
                        landmarks_dict=None,
                        quality=FaceDetectionQuality(
                            score=0.0,
                            classification="poor",
                            blur_score=0.0,
                            blur_status="blurry",
                            brightness_score=0.0,
                            brightness_status="too_dark",
                            contrast_score=0.0,
                            contrast_status="low_contrast",
                            face_area_ratio=0.0,
                            visibility="poor",
                            overall_quality=0.0,
                            sharpness=0.0,
                            brightness=0.0,
                            contrast=0.0,
                            is_usable=False,
                        ),
                        pose=None,
                    )
                    results.append(fallback_item)
                except Exception:
                    pass
        return results

    # -------------------------------------------------------------------------
    # 3. Comprehensive Structured Face Analysis (for /analyze endpoint)
    # -------------------------------------------------------------------------

    def analyze_structured_detections(
        self,
        image_rgb: np.ndarray,
        detections: List[FaceDetection],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[StructuredFaceAnalysis]:
        """
        Executes comprehensive structured analysis on detected faces, returning
        StructuredFaceAnalysis models conforming to the /api/v1/ai/analyze API endpoint.
        """
        from app.ai.analysis.quality import FaceAnalyzer
        analyzer = FaceAnalyzer(min_sharpness=self.min_sharpness)
        return analyzer.analyze_detections(image_rgb, detections)


# Singleton instance and accessor provider
_face_analysis_service_instance: Optional[FaceAnalysisService] = None


def get_face_analysis_service() -> FaceAnalysisService:
    """Dependency provider / accessor for the FaceAnalysisService singleton."""
    global _face_analysis_service_instance
    if _face_analysis_service_instance is None:
        _face_analysis_service_instance = FaceAnalysisService()
    return _face_analysis_service_instance

