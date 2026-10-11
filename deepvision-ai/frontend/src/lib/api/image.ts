/**
 * DeepVision AI — Image Detection & Facial Analysis API
 */

import { API_BASE_URL, getApiBaseUrl, ApiError } from "./client";
import {
  ImageAnalysisResponse,
  StructuredFaceAnalysis,
  BoundingBox,
  FaceLandmarks,
  LandmarkCoordinate,
} from "./types";

export interface AnalyzeImageOptions {
  confidenceThreshold?: number;
  minSharpness?: number;
}

function normalizeBoundingBox(bbox: any): BoundingBox {
  if (Array.isArray(bbox) && bbox.length >= 4) {
    const [x, y, w, h] = bbox;
    return {
      x1: x,
      y1: y,
      x2: x + w,
      y2: y + h,
      width: w,
      height: h,
    };
  }
  if (bbox && typeof bbox === "object") {
    const x1 = bbox.x1 ?? bbox.x_min ?? 0;
    const y1 = bbox.y1 ?? bbox.y_min ?? 0;
    const x2 = bbox.x2 ?? bbox.x_max ?? (bbox.width ? x1 + bbox.width : 0);
    const y2 = bbox.y2 ?? bbox.y_max ?? (bbox.height ? y1 + bbox.height : 0);
    return {
      x1,
      y1,
      x2,
      y2,
      width: bbox.width ?? Math.max(0, x2 - x1),
      height: bbox.height ?? Math.max(0, y2 - y1),
    };
  }
  return { x1: 0, y1: 0, x2: 0, y2: 0, width: 0, height: 0 };
}

function normalizeLandmark(pt: any): LandmarkCoordinate {
  if (Array.isArray(pt) && pt.length >= 2) {
    return { x: pt[0], y: pt[1] };
  }
  if (pt && typeof pt === "object") {
    return { x: pt.x ?? 0, y: pt.y ?? 0, confidence: pt.confidence };
  }
  return { x: 0, y: 0 };
}

function normalizeLandmarks(lm: any): FaceLandmarks {
  if (!lm || typeof lm !== "object") {
    return {
      right_eye: { x: 0, y: 0 },
      left_eye: { x: 0, y: 0 },
      nose_tip: { x: 0, y: 0 },
      mouth_right: { x: 0, y: 0 },
      mouth_left: { x: 0, y: 0 },
    };
  }
  return {
    right_eye: normalizeLandmark(lm.right_eye),
    left_eye: normalizeLandmark(lm.left_eye),
    nose_tip: normalizeLandmark(lm.nose_tip),
    mouth_right: normalizeLandmark(lm.mouth_right),
    mouth_left: normalizeLandmark(lm.mouth_left),
  };
}

export async function analyzeImageFile(
  file: File,
  optionsOrThreshold?: number | AnalyzeImageOptions,
  extraOptions?: AnalyzeImageOptions
): Promise<ImageAnalysisResponse> {
  let threshold: number | undefined;
  let minSharpness: number | undefined;

  if (typeof optionsOrThreshold === "number") {
    threshold = optionsOrThreshold;
    if (extraOptions) {
      minSharpness = extraOptions.minSharpness;
    }
  } else if (optionsOrThreshold && typeof optionsOrThreshold === "object") {
    threshold = optionsOrThreshold.confidenceThreshold;
    minSharpness = optionsOrThreshold.minSharpness;
  }

  const formData = new FormData();
  formData.append("file", file);

  const queryParams = new URLSearchParams();
  if (threshold !== undefined) {
    queryParams.append("confidence_threshold", threshold.toString());
  }
  if (minSharpness !== undefined) {
    queryParams.append("min_sharpness", minSharpness.toString());
  }

  const queryString = queryParams.toString() ? `?${queryParams.toString()}` : "";
  const url = `${getApiBaseUrl()}/api/v1/ai/analyze${queryString}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorMsg = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorMsg = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      } else if (errJson.message) {
        errorMsg = errJson.message;
      }
    } catch {
      // Fallback
    }
    throw new ApiError(errorMsg, response.status);
  }

  const raw = await response.json();

  // Normalize faces response to ensure StructuredFaceAnalysis format
  const faces: StructuredFaceAnalysis[] = (raw.faces || []).map((f: any, idx: number) => ({
    face_id: f.face_id ?? idx,
    face_index: f.face_index ?? idx,
    bbox: normalizeBoundingBox(f.bbox),
    confidence: f.confidence ?? 0,
    landmarks: normalizeLandmarks(f.landmarks),
    quality_score: f.quality_score ?? f.quality?.quality_score ?? 0,
    classification: f.classification ?? f.quality?.classification ?? "GOOD",
    quality: f.quality,
    clarity: f.clarity || f.blur,
    blur: f.blur || f.clarity,
    illumination: f.illumination || f.brightness,
    brightness: f.brightness || f.illumination,
    contrast: f.contrast,
    size: f.size,
    pose: f.pose || f.head_pose,
    head_pose: f.head_pose || f.pose,
  }));

  return {
    status: raw.status || "success",
    total_faces: raw.face_count ?? faces.length,
    face_count: raw.face_count ?? faces.length,
    model: raw.model || "YuNet ONNX",
    device: raw.device || "CPU",
    faces,
    image_width: raw.image_width ?? (raw.image ? raw.image.width : 0),
    image_height: raw.image_height ?? (raw.image ? raw.image.height : 0),
    processing_time_ms: raw.processing_time_ms ?? 0,
  };
}

