/**
 * DeepVision AI — Shared API & Model Schemas
 * Strictly matches backend schemas for YuNet and ByteTrack pipelines.
 */

export interface LandmarkCoordinate {
  x: number;
  y: number;
  confidence?: number;
}

export interface FaceLandmarks {
  right_eye: LandmarkCoordinate;
  left_eye: LandmarkCoordinate;
  nose_tip: LandmarkCoordinate;
  mouth_right: LandmarkCoordinate;
  mouth_left: LandmarkCoordinate;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width?: number;
  height?: number;
}

export interface HeadPoseEstimation {
  yaw: number;
  pitch: number;
  roll: number;
  pose_category?: string;
}

export type HeadPoseAngles = HeadPoseEstimation;

export interface FaceQualityScore {
  quality_score?: number;
  overall_score?: number;
  overall_quality?: number;
  clarity_score?: number;
  illumination_score?: number;
  contrast_score?: number;
  pose_score?: number;
  sharpness?: number;
  brightness?: number;
  contrast?: number;
  grade?: string;
  classification?: string;
  is_usable?: boolean;
}

export interface ClarityMetrics {
  sharpness?: number;
  variance?: number;
  blur_status?: string;
  is_sharp?: boolean;
  score?: number;
  blur_score?: number;
}

export interface IlluminationMetrics {
  brightness?: number;
  mean_brightness?: number;
  brightness_status?: string;
  is_well_lit?: boolean;
  score?: number;
  brightness_score?: number;
}

export interface ContrastMetrics {
  contrast?: number;
  michelson_contrast?: number;
  rms_contrast?: number;
  contrast_status?: string;
  score?: number;
  contrast_score?: number;
}

export interface FaceSizeMetrics {
  width?: number;
  height?: number;
  area_pixels?: number;
  size_category?: string;
}

export interface StructuredFaceAnalysis {
  face_id?: string | number;
  face_index?: number;
  bbox: BoundingBox;
  confidence: number;
  landmarks: FaceLandmarks;
  quality_score?: number;
  classification?: string;
  quality?: FaceQualityScore;
  clarity?: ClarityMetrics;
  blur?: ClarityMetrics;
  illumination?: IlluminationMetrics;
  brightness?: IlluminationMetrics;
  contrast?: ContrastMetrics;
  size?: FaceSizeMetrics;
  pose?: HeadPoseEstimation;
  head_pose?: HeadPoseEstimation;
}

export interface ImageAnalysisResponse {
  status: string;
  total_faces: number;
  face_count: number;
  model?: string;
  device?: string;
  faces: StructuredFaceAnalysis[];
  image_width: number;
  image_height: number;
  processing_time_ms: number;
}

export type VideoJobStatus =
  | "queued"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

export interface TrackSummaryItem {
  track_id: number;
  frames_active: number;
  start_frame: number;
  end_frame: number;
  avg_confidence: number;
}

export interface VideoMetrics {
  duration_seconds: number;
  total_frames: number;
  processed_frames: number;
  fps: number;
  average_processing_fps: number;
  processing_time_seconds: number;
  unique_tracks_count: number;
  peak_simultaneous_faces: number;
  tracks_summary?: TrackSummaryItem[];
}

export interface VideoJob {
  job_id: string;
  status: VideoJobStatus;
  progress: number;
  processed_frames: number;
  total_frames: number;
  current_faces?: number;
  active_tracks?: number;
  error?: string | null;
  metrics?: VideoMetrics | null;
  annotated_video_url?: string | null;
  created_at?: string;
  completed_at?: string | null;
}

export interface VideoSubmitResponse {
  job_id: string;
  status: VideoJobStatus;
  message?: string;
}

export interface SystemStatusResponse {
  status: string;
  is_ready?: boolean;
  device?: string;
  model?: string;
  tracking_algorithm?: string;
  backend_version?: string;
}
