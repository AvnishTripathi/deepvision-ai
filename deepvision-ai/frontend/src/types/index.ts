export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width?: number;
  height?: number;
  area?: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
}

export interface LandmarksDict {
  right_eye?: LandmarkPoint;
  left_eye?: LandmarkPoint;
  nose_tip?: LandmarkPoint;
  mouth_right?: LandmarkPoint;
  mouth_left?: LandmarkPoint;
  [key: string]: LandmarkPoint | undefined;
}

export interface PoseAngles {
  yaw: number;
  pitch: number;
  roll: number;
  frontality_score?: number;
  pose_category?: string;
}

export interface BlurMetrics {
  blur_score: number;
  blur_status: "sharp" | "slightly_blurred" | "blurry" | string;
  laplacian_variance?: number;
  category?: string;
}

export interface BrightnessMetrics {
  brightness_score: number;
  brightness_status: "acceptable" | "too_dark" | "too_bright" | string;
  mean_luminance?: number;
  category?: string;
}

export interface ContrastMetrics {
  contrast_score: number;
  contrast_status: "acceptable" | "low_contrast" | "high_contrast" | string;
  std_dev?: number;
}

export interface FaceSizeMetrics {
  width: number;
  height: number;
  area_fraction?: number;
  size_category?: "TINY" | "SMALL" | "MEDIUM" | "LARGE" | string;
  size_score?: number;
  is_sufficient?: boolean;
}

export interface ComprehensiveFaceQuality {
  quality_score: number;
  overall_quality?: number;
  grade: "EXCELLENT" | "GOOD" | "FAIR" | "POOR" | "UNUSABLE" | string;
  is_usable: boolean;
  classification?: string;
}

export interface StructuredFaceAnalysis {
  face_index: number;
  bbox: BoundingBox;
  confidence: number;
  quality_score?: number;
  classification?: string;
  landmarks?: LandmarksDict;
  landmarks_list?: LandmarkPoint[];
  pose?: PoseAngles | null;
  size: FaceSizeMetrics;
  illumination?: BrightnessMetrics;
  brightness?: BrightnessMetrics;
  clarity?: BlurMetrics;
  blur?: BlurMetrics;
  contrast?: ContrastMetrics;
  quality: ComprehensiveFaceQuality;
}

export interface AnalysisPerformanceBreakdown {
  total_ms: number;
  detection_ms?: number;
  landmark_ms?: number;
  quality_analysis_ms?: number;
  pose_ms?: number;
}

export interface AIAnalysisResponse {
  success: boolean;
  model: string;
  device: string;
  image_width: number;
  image_height: number;
  processing_time_ms: number;
  face_count: number;
  faces: StructuredFaceAnalysis[];
  timing?: AnalysisPerformanceBreakdown;
  detection_time_ms?: number;
  landmark_time_ms?: number;
  quality_analysis_time_ms?: number;
  pose_time_ms?: number;
  total_analysis_time_ms?: number;
  request_id?: string;
}

export interface VideoProcessingMetrics {
  total_frames: number;
  processed_frames: number;
  total_processing_time: number;
  average_frame_processing_time: number;
  average_fps: number;
  peak_simultaneous_faces: number;
  unique_track_count: number;
  track_lost_count: number;
  total_faces_detected?: number;
  detection_time_ms?: number;
  tracking_time_ms?: number;
  analysis_time_ms?: number;
  annotation_time_ms?: number;
}

export interface VideoJobResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  progress: number;
  total_frames: number;
  processed_frames: number;
  error?: string | null;
  metrics?: VideoProcessingMetrics | null;
  output_media?: string | null;
}

export interface SystemStatusResponse {
  status: string;
  model?: string;
  device?: string;
  is_ready?: boolean;
}
