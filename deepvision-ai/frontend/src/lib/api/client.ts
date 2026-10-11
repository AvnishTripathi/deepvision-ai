/**
 * DeepVision AI API Client
 * Base configuration and HTTP request helpers.
 */

export function getApiBaseUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    if (window.location.hostname === "deepvision-frontend.onrender.com") {
      return "https://deepvision-backend.onrender.com";
    }
    return window.location.origin;
  }
  if (process.env.NODE_ENV === "production") {
    return "";
  }
  return "http://localhost:8000";
}

export const API_BASE_URL = getApiBaseUrl();

export class ApiError extends Error {
  public status: number;
  public details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

export async function requestJson<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;
  const timeoutMs = options.timeoutMs ?? 30000;

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: options.signal || controller.signal,
      headers: {
        Accept: "application/json",
        ...options.headers,
      },
    });

    clearTimeout(id);

    if (!response.ok) {
      let errorMsg = `HTTP Error ${response.status}: ${response.statusText}`;
      let errorDetails: unknown = undefined;
      try {
        const errorJson = await response.json();
        errorDetails = errorJson;
        if (errorJson.detail) {
          errorMsg = typeof errorJson.detail === "string" ? errorJson.detail : JSON.stringify(errorJson.detail);
        } else if (errorJson.message) {
          errorMsg = errorJson.message;
        }
      } catch {
        // Fall back to HTTP status message
      }
      throw new ApiError(errorMsg, response.status, errorDetails);
    }

    return (await response.json()) as T;
  } catch (err: any) {
    clearTimeout(id);
    if (err.name === "AbortError") {
      throw new ApiError(`Request timeout after ${timeoutMs}ms`, 408);
    }
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      err.message || "Failed to communicate with backend server",
      0
    );
  }
}

