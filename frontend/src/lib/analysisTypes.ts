// Mirrors the endpoints added in backend/main.py for T1-T5.
// Change one, change the other -- same rule as types.ts.

// ---------------------------------------------------------------------------
// T4 -- enhancement
// ---------------------------------------------------------------------------

export interface EnhanceMetrics {
  mean: number;
  std: number;
  sharpness: number;
  speckle_index: number;
  entropy: number;
  cnr?: number;
}

export interface EnhancePanel {
  name: string;
  label: string;
  group: string;
  note?: string;
  png: string;
  metrics: EnhanceMetrics;
}

export interface EnhanceResponse {
  ok: true;
  roi: [number, number, number, number] | null;
  panels: EnhancePanel[];
  metric_notes: Record<string, string>;
}

// ---------------------------------------------------------------------------
// T4 -- compression
// ---------------------------------------------------------------------------

export interface CompressPoint {
  codec: string;
  label: string;
  lossy: boolean;
  quality: number;
  bytes: number;
  kb: number;
  ratio: number;
  bpp: number;
  /** null means lossless: MSE was zero, so PSNR is infinite. */
  psnr_db: number | null;
  ssim: number;
  mse: number;
  lossless: boolean;
}

export interface DriftPoint {
  codec: string;
  quality: number;
  ssim: number;
  psnr_db: number | null;
  area_px: number;
  area_drift_pct: number;
  echo_mean: number;
  echo_drift: number;
}

export interface CompressResponse {
  ok: true;
  codec: string;
  raw_kb: number;
  sweep: CompressPoint[];
  equal_size: (CompressPoint & { note: string })[];
  codecs: Record<string, { label: string; lossy: boolean; note: string }>;
  roi?: [number, number, number, number];
  measurement_drift?: DriftPoint[];
  examples?: { quality: number; png: string; kb: number }[];
}

// ---------------------------------------------------------------------------
// T5 -- registration
// ---------------------------------------------------------------------------

export interface RegistrationFit {
  ok: boolean;
  model: string;
  matrix: number[][] | null;
  keypoints_baseline: number;
  keypoints_followup: number;
  matches_after_ratio_test: number;
  inliers: number;
  inlier_ratio: number;
  rmse_px: number | null;
  error: string | null;
  /** Only present on the demo endpoint, where the true transform is known. */
  corner_error_px?: number;
}

export interface RegistrationViews {
  matches: string;
  warped: string;
  checkerboard: string;
  difference: string;
}

export interface RegistrationResponse extends RegistrationFit {
  views?: RegistrationViews;
  interpretation?: Record<string, string>;
}

export interface RegistrationSegmentationResponse extends RegistrationFit {
  views: RegistrationViews;
  segmentation: {
    baseline: { ok: boolean; count: number; annotated_png?: string; lesions?: Record<string, number>[] };
    followup_registered: { ok: boolean; count: number; annotated_png?: string; lesions?: Record<string, number>[] };
  };
  comparison: {
    area_change_pct: number | null;
    max_diameter_change_pct: number | null;
  } | null;
}
export interface RegistrationDemo {
  ok: true;
  true_transform: number[][];
  models: RegistrationFit[];
  baseline_png: string;
  followup_png: string;
  views: RegistrationViews | null;
  note: string;
}
