/**
 * DeepVision AI — System Status & Model Diagnostics API Client
 */

import { request } from "./client";
import { SystemStatus } from "./types";

interface RawAiStatus {
  model?: string;
  device?: string;
  status?: string;
}

interface RawHealth {
  status?: string;
}

/**
 * Checks connectivity to the FastAPI backend and retrieves active model and hardware device.
 */
export async function getSystemHealth(): Promise<SystemStatus> {
  try {
    const aiData = await request<RawAiStatus>("/ai/status");
    return {
      status: "online",
      model: aiData.model || "YuNet ONNX",
      device: aiData.device || "CPU",
      is_ready: true,
    };
  } catch {
    try {
      const healthData = await request<RawHealth>("/health");
      return {
        status: healthData.status === "healthy" || healthData.status === "ok" ? "online" : "offline",
        model: "YuNet ONNX",
        device: "CPU",
        is_ready: true,
      };
    } catch {
      return {
        status: "offline",
        is_ready: false,
      };
    }
  }
}

export const fetchSystemStatus = getSystemHealth;

