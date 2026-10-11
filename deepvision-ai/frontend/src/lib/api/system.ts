/**
 * DeepVision AI — System Telemetry & Health Probing
 */

import { getApiBaseUrl } from "./client";
import { SystemStatusResponse } from "./types";

export async function checkHealth(): Promise<boolean> {
  try {
    const baseUrl = getApiBaseUrl();
    const res = await fetch(`${baseUrl}/api/v1/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  try {
    const isHealthy = await checkHealth();
    if (!isHealthy) {
      return { status: "offline", is_ready: false };
    }

    const baseUrl = getApiBaseUrl();
    const res = await fetch(`${baseUrl}/api/v1/ai/status`, {
      cache: "no-store",
    });

    if (!res.ok) {
      return { status: "online", is_ready: true, device: "CPU", model: "YuNet ONNX" };
    }

    const data = await res.json();
    return {
      status: data.is_operational ? "online" : "ready",
      is_ready: data.is_operational,
      device: data.configured_device ? data.configured_device.toUpperCase() : "CPU",
      model: data.model_name || "YuNet ONNX",
      tracking_algorithm: "ByteTrack MOT",
      backend_version: data.model_version || "1.0.0",
    };
  } catch {
    return { status: "offline", is_ready: false };
  }
}

