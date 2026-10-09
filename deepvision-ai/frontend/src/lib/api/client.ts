/**
 * DeepVision AI — Base HTTP API Client
 * Dynamically resolves API root via NEXT_PUBLIC_API_URL or relative Next.js proxy,
 * and formats structured domain errors.
 */

export const getApiBaseUrl = (): string => {
  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  if (envUrl && envUrl.trim().length > 0) {
    // Strip trailing slash if present
    const cleanUrl = envUrl.replace(/\/+$/, "");
    return `${cleanUrl}/api/v1`;
  }
  // Fallback to Next.js route proxy
  return "/api/v1";
};

export class ApiError extends Error {
  statusCode: number;
  errorCode?: string;
  detail?: string;

  constructor(message: string, statusCode: number = 500, detail?: string, errorCode?: string) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.detail = detail;
    this.errorCode = errorCode;
  }
}

export async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        Accept: "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      let errorMessage = `Request failed with status ${response.status}`;
      let detail = "";
      let errorCode = "";

      try {
        const errorJson = await response.json();
        detail = errorJson.detail || errorJson.message || "";
        errorCode = errorJson.error_code || "";
        if (detail) {
          errorMessage = detail;
        }
      } catch {
        // Fallback to status text
        if (response.status === 413) {
          errorMessage = "File size exceeds server processing limits.";
        } else if (response.status === 400) {
          errorMessage = "Invalid input or corrupted media container.";
        } else if (response.status === 404) {
          errorMessage = "Requested resource was not found.";
        } else if (response.status === 503) {
          errorMessage = "AI model is not currently operational.";
        }
      }

      throw new ApiError(errorMessage, response.status, detail, errorCode);
    }

    return (await response.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }
    const message = err instanceof Error ? err.message : "Network failure connecting to AI backend service.";
    throw new ApiError(message, 0);
  }
}
