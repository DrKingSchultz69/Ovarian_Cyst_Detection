"""OvaScan inference core.

Pure ML + image processing. No web-framework imports live here on purpose:
this module can be exercised from a plain Python REPL, which makes debugging
segmentation problems far faster than debugging them through HTTP.

burnin_mask(), preprocess() and the measurement maths are ported unchanged
from the validated Kaggle notebook so results match what was measured there.
The new surface is analyse_bytes(), which maps raw image bytes onto the JSON
contract the frontend consumes.
"""

from __future__ import annotations

import base64
import glob
import os
import time
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Weights resolution
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.join(_HERE, "weights")
_DEFAULT_WEIGHTS = os.path.join(_WEIGHTS_DIR, "ovascan_burnin1.pt")

# Bytes; anything larger is rejected before it reaches the model.
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# This model ships a single class.
CLASS_NAME = "lesion"


def resolve_weights() -> str:
    """Locate the .pt file: OVASCAN_WEIGHTS wins, then the default filename,
    then the newest .pt anywhere under weights/."""
    env = os.environ.get("OVASCAN_WEIGHTS")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"OVASCAN_WEIGHTS points at a missing file: {env}")
        return env

    # Upload UIs that flatten folders (the HF Spaces uploader does) drop the
    # checkpoint next to the source instead of under weights/, so check both.
    for candidate in (_DEFAULT_WEIGHTS, os.path.join(_HERE, "ovascan_burnin1.pt")):
        if os.path.exists(candidate):
            return candidate

    found = sorted(
        glob.glob(os.path.join(_WEIGHTS_DIR, "**", "*.pt"), recursive=True)
        + glob.glob(os.path.join(_HERE, "*.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not found:
        raise FileNotFoundError(
            "No model weights found. Put ovascan_burnin1.pt in "
            f"{_WEIGHTS_DIR} or beside inference.py, or set OVASCAN_WEIGHTS."
        )
    return found[0]


# ---------------------------------------------------------------------------
# Model: loaded once at import, warmed once. Never load inside a request
# handler -- that re-reads the checkpoint and re-initialises torch every call.
# ---------------------------------------------------------------------------

WEIGHTS_PATH = resolve_weights()
model = YOLO(WEIGHTS_PATH)
model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)  # warm-up


def model_info() -> dict[str, Any]:
    names = getattr(model, "names", None) or {0: CLASS_NAME}
    return {
        "weights": os.path.basename(WEIGHTS_PATH),
        "weights_path": WEIGHTS_PATH,
        "classes": list(names.values()),
    }


# ---------------------------------------------------------------------------
# Burn-in removal
#   Scanner UI text and calipers are high-saturation or near-white. Keep only
#   the small, compact components (real anatomy is neither), dilate a little so
#   the fill covers antialiased edges, then Telea-inpaint.
# ---------------------------------------------------------------------------

def burnin_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    raw = ((hsv[:, :, 1] > 60) | (gray > 235)).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = np.zeros_like(raw)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        compactness = area / float(max(w * h, 1))  # rejects long sparse streaks
        if 3 < area < 1200 and compactness > 0.15:
            keep[lab == i] = 1

    return cv2.dilate(keep, np.ones((5, 5), np.uint8), 1) * 255


def preprocess(bgr: np.ndarray) -> np.ndarray:
    return cv2.inpaint(bgr, burnin_mask(bgr), 3, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Measurements -- every value is in image pixels. Converting to mm needs the
# scanner scale bar or DICOM PixelSpacing, which this prototype does not read.
# ---------------------------------------------------------------------------

def measure(mask_u8: np.ndarray, gray: np.ndarray) -> Optional[dict[str, float]]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = float((mask_u8 > 0).sum())
    perimeter = float(cv2.arcLength(c, True))

    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) or 1.0

    # Max caliper: largest pairwise distance across convex hull vertices.
    hp = hull.reshape(-1, 2).astype(np.float32)
    max_diameter = float(np.sqrt(((hp[:, None, :] - hp[None, :, :]) ** 2).sum(-1)).max())

    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    px = gray[mask_u8 > 0]

    return {
        "max_diameter_px": round(max_diameter, 1),
        "area_px": int(area),
        "circularity": round(float(min(circularity, 1.0)), 3),
        "solidity": round(area / hull_area, 3),
        "echo_mean": round(float(px.mean()), 1),
        "echo_sd": round(float(px.std()), 1),
    }


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Not a decodable image (expected jpg / png / bmp / tif).")
    return bgr


# ---------------------------------------------------------------------------
# Annotation: translucent mask fill, bright boundary, box + confidence label
# ---------------------------------------------------------------------------

_FILL = (255, 190, 0)   # BGR, amber
_EDGE = (0, 255, 120)   # BGR, bright green
_FILL_ALPHA = 0.35


def _annotate(base_bgr: np.ndarray, polys, boxes_xyxy, confs) -> np.ndarray:
    out = base_bgr.copy()

    overlay = out.copy()
    for poly in polys:
        cv2.fillPoly(overlay, [poly], _FILL)
    out = cv2.addWeighted(overlay, _FILL_ALPHA, out, 1 - _FILL_ALPHA, 0)

    for poly, (x1, y1, x2, y2), conf in zip(polys, boxes_xyxy, confs):
        cv2.polylines(out, [poly], True, _EDGE, 2, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), _EDGE, 2)

        label = f"{CLASS_NAME} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), _EDGE, -1)
        cv2.putText(out, label, (x1 + 3, max(y1 - 5, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Public entry point -- returns the API contract as a plain dict.
# Never raises: failures come back as {"ok": False, "error": ...} so a bad
# upload degrades to a message in the UI instead of a 500.
# ---------------------------------------------------------------------------

def analyse_bytes(image_bytes: bytes, conf: float = 0.25,
                  include_debug: bool = False) -> dict[str, Any]:
    started = time.time()
    try:
        if not image_bytes:
            raise ValueError("Empty request body.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image is {len(image_bytes) / 1e6:.1f} MB; limit is "
                f"{MAX_IMAGE_BYTES / 1e6:.0f} MB."
            )

        conf = float(np.clip(conf, 0.05, 0.95))
        bgr = _decode(image_bytes)
        clean = preprocess(bgr)                      # burn-in removed pre-inference
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        h, w = clean.shape[:2]

        result = model.predict(clean, conf=conf, verbose=False)[0]

        payload: dict[str, Any] = {
            "ok": True,
            "conf": round(conf, 2),
            "image_size": {"width": w, "height": h},
            "units": "pixels",
            "note": ("All distances and areas are in image pixels. Conversion to mm "
                     "requires the scanner scale bar or DICOM pixel spacing."),
        }

        if include_debug:
            payload["debug"] = {
                "raw_png": _png_b64(bgr),
                "burnin_mask_png": _png_b64(
                    cv2.applyColorMap(burnin_mask(bgr), cv2.COLORMAP_HOT)),
                "inpainted_png": _png_b64(clean),
            }

        if result.masks is None or len(result.boxes) == 0:
            annotated = clean.copy()
            cv2.putText(annotated, "No lesion detected", (14, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            payload.update({
                "count": 0,
                "lesions": [],
                "annotated_png": _png_b64(annotated),
                "latency_ms": int((time.time() - started) * 1000),
            })
            return payload

        polys, boxes, confs, lesions = [], [], [], []
        for i, poly in enumerate(result.masks.xy):
            if len(poly) < 3:
                continue
            poly_i = poly.astype(np.int32)

            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [poly_i], 255)
            stats = measure(mask, gray)
            if stats is None:
                continue

            x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            confidence = float(result.boxes.conf[i])

            polys.append(poly_i)
            boxes.append((x1, y1, x2, y2))
            confs.append(confidence)
            lesions.append({
                "id": len(lesions) + 1,
                "conf": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
                **stats,
            })

        payload.update({
            "count": len(lesions),
            "lesions": lesions,
            "annotated_png": _png_b64(_annotate(clean, polys, boxes, confs)),
            "latency_ms": int((time.time() - started) * 1000),
        })
        return payload

    except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the boundary
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "count": 0,
            "lesions": [],
            "annotated_png": None,
            "latency_ms": int((time.time() - started) * 1000),
        }
