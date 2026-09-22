// Mirrors the API contract in backend/README.md. Change one, change the other.

export interface Lesion {
  id: number;
  conf: number;
  bbox: [number, number, number, number];
  max_diameter_px: number;
  area_px: number;
  circularity: number;
  solidity: number;
  echo_mean: number;
  echo_sd: number;
}

export interface PredictDebug {
  raw_png: string;
  burnin_mask_png: string;
  inpainted_png: string;
}

export interface PredictSuccess {
  ok: true;
  conf: number;
  count: number;
  latency_ms: number;
  units: string;
  note: string;
  image_size: { width: number; height: number };
  annotated_png: string;
  lesions: Lesion[];
  debug?: PredictDebug;
}

export interface PredictFailure {
  ok: false;
  error: string;
  count: 0;
  lesions: [];
  annotated_png: null;
  latency_ms: number;
}

export type PredictResponse = PredictSuccess | PredictFailure;

/** count === 0 on a successful call is a valid negative result, not an error. */
export const isNegative = (r: PredictResponse): boolean => r.ok && r.count === 0;
