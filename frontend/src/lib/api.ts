// Tiny client for the FastAPI backend. Override the base URL with
// NEXT_PUBLIC_API_URL (e.g. in frontend/.env.local); defaults to localhost:8000.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface ModelInfo {
  backbone: string;
  checkpoint: string | null;
  device: string;
  img_size: number;
  random_weights: boolean;
  vlm_enabled: boolean;
}

export interface PredictResponse {
  label: "good" | "defect";
  confidence: number;
  defect_probability: number;
  heatmap_png_b64: string;
  description: string | null;
  model: ModelInfo;
}

export interface HealthResponse {
  status: string;
  model_ready: boolean;
  model: ModelInfo | null;
  error: string | null;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`health check failed (${res.status})`);
  return res.json();
}

export async function predict(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let detail = `request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body — keep the status message */
    }
    throw new Error(detail);
  }
  return res.json();
}
