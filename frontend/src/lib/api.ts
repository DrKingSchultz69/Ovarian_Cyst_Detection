import { MOCK_EMPTY, MOCK_RESPONSE } from "./mock";
import type { PredictResponse } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");

/** No API URL configured means Phase 2 mode: the UI runs off the mock. */
export const usingMock = !API;

export const apiBase = API ?? null;

// ngrok's free tier intercepts browser requests with an HTML interstitial, so
// fetch() gets that page instead of JSON while /docs looks perfectly fine.
// This header opts out. Harmless against any other host.
const NGROK_HEADER = { "ngrok-skip-browser-warning": "true" } as const;

export async function predict(
  file: File,
  conf: number,
  debug: boolean,
): Promise<PredictResponse> {
  if (!API) {
    await new Promise((r) => setTimeout(r, 700)); // make the loading state visible
    // Filename escape hatch so the empty state can be demoed without a backend.
    return /empty|none|negative/i.test(file.name) ? MOCK_EMPTY : MOCK_RESPONSE;
  }

  const body = new FormData();
  body.append("file", file);
  body.append("conf", String(conf));
  body.append("debug", String(debug));

  try {
    const res = await fetch(`${API}/predict`, { method: "POST", body, headers: NGROK_HEADER });

    // 422 still carries a usable JSON body; 413/415 carry FastAPI's `detail`.
    const text = await res.text();
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return fail(`Server returned non-JSON (HTTP ${res.status}): ${text.slice(0, 160)}`);
    }

    const data = parsed as Record<string, unknown>;
    if (typeof data.ok === "boolean") return data as unknown as PredictResponse;
    if (typeof data.detail === "string") return fail(data.detail);
    return fail(`Unexpected response shape (HTTP ${res.status}).`);
  } catch {
    // fetch() rejects on network failure and on CORS rejection alike, and the
    // browser deliberately hides which. Both are covered by this message.
    return fail(
      `Could not reach the API at ${API}. Check the service is awake and that ` +
        `its CORS origins allow this page.`,
    );
  }
}

export async function health(): Promise<boolean> {
  if (!API) return true;
  try {
    const res = await fetch(`${API}/health`, { cache: "no-store", headers: NGROK_HEADER });
    return res.ok;
  } catch {
    return false;
  }
}

function fail(error: string): PredictResponse {
  return { ok: false, error, count: 0, lesions: [], annotated_png: null, latency_ms: 0 };
}
