/**
 * DeepVision AI — Video Processing & ByteTrack Multi-Face Tracking API Client
 */

import { request, getApiBaseUrl } from "./client";
import { VideoJob } from "./types";

export interface VideoProcessingSubmitOptions {
  confidenceThreshold?: number;
  annotateVideo?: boolean;
  detectionInterval?: number;
  analysisInterval?: number;
}

/**
 * Submits a video file to the asynchronous tracking engine.
 */
export async function submitVideoAnalysisJob(
  file: File,
  options: VideoProcessingSubmitOptions = {}
): Promise<VideoJob> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  if (options.confidenceThreshold !== undefined) {
    params.set("confidence_threshold", options.confidenceThreshold.toString());
  }
  if (options.annotateVideo !== undefined) {
    params.set("annotate_video", options.annotateVideo ? "true" : "false");
  }
  if (options.detectionInterval !== undefined) {
    params.set("detection_interval", options.detectionInterval.toString());
  }
  if (options.analysisInterval !== undefined) {
    params.set("analysis_interval", options.analysisInterval.toString());
  }

  const query = params.toString() ? `?${params.toString()}` : "";
  const endpoint = `/video/analyze${query}`;

  return request<VideoJob>(endpoint, {
    method: "POST",
    body: formData,
  });
}

/**
 * Polls the current execution status, frame count, progress percentage,
 * and telemetry metrics for a video processing job.
 */
export async function pollVideoJobStatus(jobId: string): Promise<VideoJob> {
  return request<VideoJob>(`/video/jobs/${jobId}`, {
    method: "GET",
  });
}

/**
 * Requests graceful cancellation of an active video processing job.
 */
export async function cancelVideoJobExecution(jobId: string): Promise<VideoJob> {
  return request<VideoJob>(`/video/jobs/${jobId}/cancel`, {
    method: "POST",
  });
}

/**
 * Constructs absolute URL for downloading the annotated MP4 video artifact.
 */
export function buildVideoDownloadUrl(jobId: string): string {
  const baseUrl = getApiBaseUrl();
  return `${baseUrl}/video/jobs/${jobId}/download`;
}
