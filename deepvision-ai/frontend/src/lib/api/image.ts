/**
 * DeepVision AI — Image Detection & Face Analysis API Client
 */

import { request } from "./client";
import { ImageAnalysisResponse } from "./types";

export interface ImageAnalysisOptions {
  confidenceThreshold?: number;
}

/**
 * Uploads an image file and executes full face detection, landmark localization,
 * 3D head pose estimation, and composite quality grading.
 */
export async function analyzeImageFile(
  file: File,
  options: ImageAnalysisOptions = {}
): Promise<ImageAnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const threshold = options.confidenceThreshold ?? 0.5;
  const endpoint = `/ai/analyze?confidence_threshold=${threshold}`;

  return request<ImageAnalysisResponse>(endpoint, {
    method: "POST",
    body: formData,
  });
}
