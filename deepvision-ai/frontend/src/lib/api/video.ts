/**
 * DeepVision AI — Asynchronous Video Tracking & MOT API
 */

import { API_BASE_URL, ApiError } from "./client";
import { VideoJob, VideoSubmitResponse } from "./types";

export interface VideoProcessingRequestOptions {
  confidenceThreshold?: number;
  detectionInterval?: number;
  analysisInterval?: number;
  annotateVideo?: boolean;
  enableAnalysis?: boolean;
}

export async function submitVideoAnalysisJob(
  file: File,
  options: VideoProcessingRequestOptions = {}
): Promise<VideoSubmitResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const queryParams = new URLSearchParams();
  if (options.confidenceThreshold !== undefined) {
    queryParams.append("confidence_threshold", options.confidenceThreshold.toString());
  }
  if (options.detectionInterval !== undefined) {
    queryParams.append("detection_interval", options.detectionInterval.toString());
  }
  if (options.analysisInterval !== undefined) {
    queryParams.append("analysis_interval", options.analysisInterval.toString());
  }
  if (options.annotateVideo !== undefined) {
    queryParams.append("annotate_video", options.annotateVideo.toString());
  }
  if (options.enableAnalysis !== undefined) {
    queryParams.append("enable_analysis", options.enableAnalysis.toString());
  }

  const queryString = queryParams.toString() ? `?${queryParams.toString()}` : "";
  const url = `${API_BASE_URL}/api/v1/video/analyze${queryString}`;

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

  return (await response.json()) as VideoSubmitResponse;
}

export async function pollVideoJobStatus(jobId: string): Promise<VideoJob> {
  const url = `${API_BASE_URL}/api/v1/video/jobs/${encodeURIComponent(jobId)}`;
  const response = await fetch(url);

  if (!response.ok) {
    let errorMsg = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorMsg = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Fallback
    }
    throw new ApiError(errorMsg, response.status);
  }

  const raw = await response.json();
  const annotatedVideoUrl =
    raw.status === "completed"
      ? `${API_BASE_URL}/api/v1/video/jobs/${encodeURIComponent(jobId)}/download`
      : null;

  return {
    job_id: raw.job_id,
    status: raw.status,
    progress: raw.progress ?? 0,
    processed_frames: raw.processed_frames ?? 0,
    total_frames: raw.total_frames ?? 0,
    current_faces: raw.current_faces,
    active_tracks: raw.active_tracks,
    error: raw.error ?? null,
    metrics: raw.metrics ?? null,
    annotated_video_url: annotatedVideoUrl,
    created_at: raw.created_at,
    completed_at: raw.completed_at,
  };
}

export async function cancelVideoJobExecution(jobId: string): Promise<boolean> {
  const url = `${API_BASE_URL}/api/v1/video/jobs/${encodeURIComponent(jobId)}/cancel`;
  try {
    const response = await fetch(url, { method: "POST" });
    return response.ok;
  } catch {
    return false;
  }
}

export function getVideoDownloadUrl(jobId: string): string {
  return `${API_BASE_URL}/api/v1/video/jobs/${encodeURIComponent(jobId)}/download`;
}

