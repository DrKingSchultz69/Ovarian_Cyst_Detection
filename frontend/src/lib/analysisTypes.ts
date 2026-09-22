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

// ---------------------------------------------------------------------------
// T1-T3 -- EHR
// ---------------------------------------------------------------------------

export interface Icd10Row {
  code: string;
  display: string;
  local_synonyms: string[];
}

export interface LoincRow {
  code: string;
  display: string;
  canonical_unit: string;
  plausible_range: [number, number];
  local_names: string[];
}

export interface CleanFlag {
  rule: string;
  table: string;
  row: number;
  column: string;
  value: number | string | null;
  action: "quarantined" | "flagged";
  detail: string;
}

export interface MatchPair {
  left_id: string;
  right_id: string;
  left_name: string;
  right_name: string;
  birth_date: string | null;
  score: number;
}

export interface MissingProfileRow {
  loinc: string;
  display: string;
  n: number;
  missing: number;
  missing_pct: number;
  mean_observed: number | null;
}

export interface ImputationRow {
  observation: string;
  loinc: string;
  strategy: "mean" | "median" | "knn" | "iterative";
  n_imputed: number;
  rmse: number;
  mae: number;
  bias: number;
  true_mean: number;
  imputed_mean: number;
}

export interface DedupeEvaluation {
  true_duplicates_injected: number;
  pairs_found: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1: number;
  missed_pairs: string[][];
}

export interface EhrResponse {
  ok: true;
  params: { n_patients: number; seed: number };
  source: {
    patients: number;
    encounters: number;
    observations: number;
    imaging_studies: number;
    injected_defects: Record<string, number>;
  };
  coding_systems: { icd10: Icd10Row[]; loinc: LoincRow[]; note: string };
  standardize: Record<string, Record<string, unknown>>;
  clean: {
    flags_raised: number;
    values_quarantined: number;
    values_flagged_only: number;
    by_rule?: Record<string, number>;
  };
  flags: CleanFlag[];
  dedupe: Record<string, unknown>;
  dedupe_evaluation: DedupeEvaluation;
  match_pairs: MatchPair[];
  missing: {
    encounters: number;
    complete_rows: number;
    complete_case_loss: string;
    cells_missing: number;
    declared_mechanisms: Record<string, string>;
  };
  missing_profile: MissingProfileRow[];
  mechanism_diagnostics: Record<string, { against: string; correlation: number }[]>;
  imputation_comparison: ImputationRow[];
  samples: Record<string, Record<string, unknown>[]>;
}
