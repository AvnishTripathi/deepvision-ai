import {
  AIAnalysisResponse,
  VideoJobResponse,
  SystemStatusResponse,
} from "@/types";

const API_BASE = "/api/v1";

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  try {
    const res = await fetch(`${API_BASE}/ai/status`);
    if (!res.ok) {
      // Fallback to health endpoint
      const healthRes = await fetch(`${API_BASE}/health`);
      if (healthRes.ok) {
        return { status: "online", is_ready: true };
      }
      return { status: "offline", is_ready: false };
    }
    const data = await res.json();
    return {
      status: "online",
      model: data.model || "YuNet ONNX",
      device: data.device || "CPU",
      is_ready: true,
    };
  } catch (err) {
    return { status: "offline", is_ready: false };
  }
}

export async function analyzeImage(
  file: File,
  confidenceThreshold: number = 0.5
): Promise<AIAnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API_BASE}/ai/analyze?confidence_threshold=${confidenceThreshold}`;
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = "Failed to analyze image";
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch {
      // Keep default
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function submitVideoJob(
  file: File,
  options?: {
    confidenceThreshold?: number;
    annotateVideo?: boolean;
    detectionInterval?: number;
    analysisInterval?: number;
  }
): Promise<VideoJobResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  if (options?.confidenceThreshold !== undefined) {
    params.set("confidence_threshold", options.confidenceThreshold.toString());
  }
  if (options?.annotateVideo !== undefined) {
    params.set("annotate_video", options.annotateVideo ? "true" : "false");
  }
  if (options?.detectionInterval !== undefined) {
    params.set("detection_interval", options.detectionInterval.toString());
  }
  if (options?.analysisInterval !== undefined) {
    params.set("analysis_interval", options.analysisInterval.toString());
  }

  const queryString = params.toString() ? `?${params.toString()}` : "";
  const url = `${API_BASE}/video/analyze${queryString}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = "Failed to submit video processing job";
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch {
      // Keep default
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function getVideoJobStatus(jobId: string): Promise<VideoJobResponse> {
  const response = await fetch(`${API_BASE}/video/jobs/${jobId}`);
  if (!response.ok) {
    let errorDetail = "Failed to fetch job status";
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errorDetail;
    } catch {
      // Keep default
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export async function cancelVideoJob(jobId: string): Promise<VideoJobResponse> {
  const response = await fetch(`${API_BASE}/video/jobs/${jobId}/cancel`, {
    method: "POST",
  });
  if (!response.ok) {
    let errorDetail = "Failed to cancel video job";
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errorDetail;
    } catch {
      // Keep default
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export function getVideoDownloadUrl(jobId: string): string {
  return `${API_BASE}/video/jobs/${jobId}/download`;
}
