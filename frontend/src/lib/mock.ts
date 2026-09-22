// Stand-in for the API while the model trains. Phase 3 deletes this file and
// the import in api.ts; nothing else in the UI knows the difference.

import { MOCK_ANNOTATED_PNG } from "./mockImage";
import type { PredictSuccess } from "./types";

export const MOCK_RESPONSE: PredictSuccess = {
  ok: true,
  conf: 0.25,
  count: 2,
  latency_ms: 341,
  units: "pixels",
  note:
    "All distances and areas are in image pixels. Conversion to mm requires " +
    "the scanner scale bar or DICOM pixel spacing.",
  image_size: { width: 640, height: 480 },
  annotated_png: MOCK_ANNOTATED_PNG,
  lesions: [
    {
      id: 1,
      conf: 0.871,
      bbox: [146, 128, 354, 292],
      max_diameter_px: 192.4,
      area_px: 22318,
      circularity: 0.918,
      solidity: 0.971,
      echo_mean: 31.2,
      echo_sd: 6.4,
    },
    {
      id: 2,
      conf: 0.402,
      bbox: [402, 268, 498, 344],
      max_diameter_px: 96.1,
      area_px: 5734,
      circularity: 0.784,
      solidity: 0.933,
      echo_mean: 58.7,
      echo_sd: 14.9,
    },
  ],
  debug: {
    raw_png: MOCK_ANNOTATED_PNG,
    burnin_mask_png: MOCK_ANNOTATED_PNG,
    inpainted_png: MOCK_ANNOTATED_PNG,
  },
};

export const MOCK_EMPTY: PredictSuccess = {
  ...MOCK_RESPONSE,
  count: 0,
  lesions: [],
  latency_ms: 288,
};
