/** Client for the T1-T5 endpoints.
 *
 *  Separate from api.ts on purpose: api.ts owns /predict and its mock, and
 *  these endpoints have no mock. They need a real backend, and they say so
 *  plainly rather than silently returning fixtures -- a mocked EHR pipeline
 *  would be a mock of a mock, which is not worth the confusion.
 */

import type {
  CompressResponse,
  EhrResponse,
  EnhanceResponse,
  RegistrationDemo,
  RegistrationResponse,
  RegistrationSegmentationResponse,
} from "./analysisTypes";

const API = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");

export const analysisAvailable = Boolean(API);
export const analysisBase = API ?? null;

// Same ngrok interstitial opt-out as api.ts. Harmless against other hosts.
const HEADERS = { "ngrok-skip-browser-warning": "true" } as const;

export class ApiUnavailable extends Error {
  constructor() {
    super(
      "No NEXT_PUBLIC_API_URL is set, so there is no backend to run this against. " +
        "Start the API and set the variable, then rebuild — NEXT_PUBLIC_* is " +
        "inlined at build time, so a restart alone is not enough.",
    );
    this.name = "ApiUnavailable";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API) throw new ApiUnavailable();

  let response: Response;
  try {
    response = await fetch(`${API}${path}`, { ...init, headers: HEADERS });
  } catch {
    // fetch() rejects identically for a network failure and a CORS rejection,
    // and the browser deliberately hides which. Cover both.
    throw new Error(
      `Could not reach ${API}. Check the service is awake and that its ` +
        `CORS origins allow this page.`,
    );
  }

  const text = await response.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(
      `Server returned non-JSON (HTTP ${response.status}): ${text.slice(0, 160)}`,
    );
  }

  if (!response.ok) {
    const detail = (parsed as { detail?: string })?.detail;
    throw new Error(detail ?? `HTTP ${response.status}`);
  }
  return parsed as T;
}

// ---------------------------------------------------------------------------
// T4
// ---------------------------------------------------------------------------

export function enhanceImage(
  file: File,
  opts: { methods?: string[]; roi?: [number, number, number, number] } = {},
): Promise<EnhanceResponse> {
  const body = new FormData();
  body.append("file", file);
  body.append("methods", (opts.methods ?? []).join(","));
  body.append("roi", opts.roi ? opts.roi.join(",") : "");
  return request<EnhanceResponse>("/image/enhance", { method: "POST", body });
}

export function compressImage(
  file: File,
  opts: {
    codec?: string;
    roi?: [number, number, number, number];
    targetKb?: number;
  } = {},
): Promise<CompressResponse> {
  const body = new FormData();
  body.append("file", file);
  body.append("codec", opts.codec ?? "jpeg");
  body.append("roi", opts.roi ? opts.roi.join(",") : "");
  body.append("target_kb", String(opts.targetKb ?? 40));
  return request<CompressResponse>("/image/compress", { method: "POST", body });
}

// ---------------------------------------------------------------------------
// T5
// ---------------------------------------------------------------------------

export function registerImages(
  baseline: File,
  followUp: File,
  opts: { model?: string; ratio?: number; contrastThreshold?: number } = {},
): Promise<RegistrationResponse> {
  const body = new FormData();
  body.append("baseline", baseline);
  body.append("follow_up", followUp);
  body.append("model", opts.model ?? "affine");
  body.append("ratio", String(opts.ratio ?? 0.72));
  body.append("contrast_threshold", String(opts.contrastThreshold ?? 0.04));
  return request<RegistrationResponse>("/image/register", { method: "POST", body });
}

export function registrationDemo(
  opts: { seed?: number; rotation?: number; scale?: number } = {},
): Promise<RegistrationDemo> {
  const query = new URLSearchParams({
    seed: String(opts.seed ?? 0),
    rotation: String(opts.rotation ?? 6),
    scale: String(opts.scale ?? 1.06),
  });
  return request<RegistrationDemo>(`/image/register/demo?${query}`);
}

// ---------------------------------------------------------------------------
// Phantom + EHR
// ---------------------------------------------------------------------------

export function registerAndSegment(
  baseline: File,
  followUp: File,
  opts: { model?: string; conf?: number } = {},
): Promise<RegistrationSegmentationResponse> {
  const body = new FormData();
  body.append("baseline", baseline);
  body.append("follow_up", followUp);
  body.append("model", opts.model ?? "affine");
  body.append("conf", String(opts.conf ?? 0.25));
  return request<RegistrationSegmentationResponse>("/image/register-and-segment", { method: "POST", body });
}
export interface PhantomResponse {
  ok: true;
  png: string;
  lesion_bbox: [number, number, number, number];
  width: number;
  height: number;
  note: string;
}

export function getPhantom(seed = 0): Promise<PhantomResponse> {
  return request<PhantomResponse>(`/image/phantom?seed=${seed}`);
}

export function getEhrPipeline(
  opts: { nPatients?: number; seed?: number; rows?: number } = {},
): Promise<EhrResponse> {
  const query = new URLSearchParams({
    n_patients: String(opts.nPatients ?? 400),
    seed: String(opts.seed ?? 7),
    rows: String(opts.rows ?? 25),
  });
  return request<EhrResponse>(`/ehr/pipeline?${query}`);
}

/** Turn a base64 PNG from any endpoint into a usable <img src>. */
export const pngSrc = (b64: string) => `data:image/png;base64,${b64}`;

/** Fetch the phantom and hand it back as a File, so the enhance/compress
 *  endpoints can be driven without the user having an ultrasound to hand. */
export async function phantomAsFile(seed = 0): Promise<{
  file: File;
  bbox: [number, number, number, number];
}> {
  const phantom = await getPhantom(seed);
  const bytes = Uint8Array.from(atob(phantom.png), (c) => c.charCodeAt(0));
  const file = new File([bytes], `phantom-${seed}.png`, { type: "image/png" });
  return { file, bbox: phantom.lesion_bbox };
}
