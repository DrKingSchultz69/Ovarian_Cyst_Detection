"""T4 part two -- image compression.

The medical-imaging question is not "how small can it get" but "how small can
it get before it stops being diagnostic", and those have different answers for
different tasks. This module measures the trade-off rather than asserting it.

LOSSLESS vs LOSSY
    Lossless (PNG, and the lossless JPEG-2000 / JPEG-LS modes DICOM actually
    specifies) reconstructs the original bit for bit. Typical ultrasound ratio
    is only about 2-3x, because speckle is high-entropy and nearly
    incompressible -- it looks like noise to an entropy coder, which is
    precisely what it is.

    Lossy (JPEG, WebP) reaches 10-50x by discarding high-frequency detail. On
    ultrasound that detail IS the speckle, so the image often looks *cleaner*
    after compression while measurements drift.

WHY THIS IS NOT A FREE LUNCH
    A pretty image that segments differently is worse than an ugly one that
    does not. So this module reports two different things:

      fidelity   PSNR / SSIM against the original -- how close the pixels are
      task       how far the LESION MEASUREMENTS move -- how close the answer is

    They disagree, and the disagreement is the point: fidelity degrades far
    faster than the measurement does. On the synthetic phantom, JPEG at
    quality 20 drops SSIM to 0.855 while the lesion area moves only -0.7%.

    That gap is a property of THIS target, not a general licence. The phantom
    lesion is a large, high-contrast, anechoic region with a sharp wall --
    the easiest possible segmentation case. A small isoechoic lesion with a
    diffuse border sits much closer to the detail JPEG discards first, and
    would be expected to drift considerably more. Re-run measurement_drift()
    on real images before drawing any conclusion from the phantom numbers.

    A SECOND, SHARPER FINDING
    SSIM is NOT monotonic in JPEG quality on these images -- measured on the
    phantom it goes 0.906 at q=70 but 0.933 at q=60, and 0.869 at q=40 but
    0.890 at q=30, while PSNR falls monotonically throughout. This is not a
    bug. SSIM's structure term compares local variance patterns, and in a
    speckle-dominated image the local variance is itself close to random, so
    whether a given quantisation table happens to preserve one speckle
    realisation is largely luck. It is direct evidence for the caveat under
    `fidelity()`: SSIM was built for natural images and ultrasound violates
    its assumptions. Quote PSNR alongside it, and prefer the task metric to
    either.

REGULATORY NOTE
    Lossy compression of diagnostic images is restricted in most
    jurisdictions, and where it is allowed the ratio is capped by modality and
    must be disclosed. Nothing here is a recommendation to use it clinically.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# The codecs OpenCV can write in-memory. Quality means different things to
# each, which is exactly why comparing them at "quality 80" is meaningless and
# comparing them at equal SIZE is not.
CODECS: dict[str, dict[str, object]] = {
    "png": {
        "ext": ".png", "lossy": False, "label": "PNG",
        "param": cv2.IMWRITE_PNG_COMPRESSION, "range": (0, 9),
        "note": "Lossless. DEFLATE; the parameter trades CPU for size, never quality.",
    },
    "jpeg": {
        "ext": ".jpg", "lossy": True, "label": "JPEG",
        "param": cv2.IMWRITE_JPEG_QUALITY, "range": (1, 100),
        "note": "Lossy. 8x8 DCT; blocking artifacts appear first at edges.",
    },
    "webp": {
        "ext": ".webp", "lossy": True, "label": "WebP",
        "param": cv2.IMWRITE_WEBP_QUALITY, "range": (1, 100),
        "note": "Lossy. Better rate-distortion than JPEG at the same size.",
    },
}


def encode(image: np.ndarray, codec: str, quality: int) -> bytes:
    """Compress in memory. No temp files -- this runs inside a request."""
    spec = CODECS[codec]
    ok, buffer = cv2.imencode(str(spec["ext"]), image, [int(spec["param"]), int(quality)])
    if not ok:
        raise RuntimeError(f"{codec} encoding failed at quality {quality}")
    return buffer.tobytes()


def decode(payload: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("decode failed")
    return decoded


def fidelity(original: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    """Pixel-fidelity metrics against the original.

      PSNR  log-scaled mean squared error, in dB. Above ~40 dB differences are
            usually invisible; it correlates poorly with perceived quality
            because it treats every pixel independently.
      SSIM  compares local luminance, contrast and structure. Closer to human
            judgement, which is why it is the one to quote -- but it was
            designed for natural images, and speckle violates its assumptions,
            so on ultrasound it reads slightly optimistic.
    """
    a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
    b = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY) if restored.ndim == 3 else restored

    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

    # A lossless codec reconstructs exactly, so MSE is 0 and PSNR is infinite.
    # Reported as None rather than float('inf'): infinity is not JSON
    # encodable, and "perfect" is the honest reading anyway. Callers must
    # treat None as lossless, not as missing.
    return {
        "psnr_db": round(float(peak_signal_noise_ratio(a, b, data_range=255)), 2)
        if mse > 0 else None,
        "ssim": round(float(structural_similarity(a, b, data_range=255)), 5),
        "mse": round(mse, 3),
        "lossless": mse == 0.0,
    }


def sweep(image: np.ndarray, codec: str = "jpeg",
          qualities: list[int] | None = None) -> list[dict[str, object]]:
    """Rate-distortion curve: size, ratio, PSNR and SSIM at each quality.

    The raw baseline is the uncompressed byte count (h * w * channels), not
    the size of whatever file the image arrived in -- comparing against an
    already-compressed source measures the wrong thing.
    """
    spec = CODECS[codec]
    if qualities is None:
        qualities = ([0, 3, 6, 9] if not spec["lossy"]
                     else [95, 90, 80, 70, 60, 50, 40, 30, 20, 10])

    raw_bytes = int(image.nbytes)
    rows: list[dict[str, object]] = []

    for quality in qualities:
        payload = encode(image, codec, quality)
        restored = decode(payload)
        rows.append({
            "codec": codec,
            "label": spec["label"],
            "lossy": spec["lossy"],
            "quality": quality,
            "bytes": len(payload),
            "kb": round(len(payload) / 1024, 1),
            "ratio": round(raw_bytes / max(len(payload), 1), 2),
            "bpp": round(8 * len(payload) / (image.shape[0] * image.shape[1]), 3),
            **fidelity(image, restored),
        })
    return rows


def measurement_drift(image: np.ndarray, roi: tuple[int, int, int, int],
                      codec: str = "jpeg",
                      qualities: list[int] | None = None) -> list[dict[str, object]]:
    """How far the MEASUREMENTS move under compression.

    Uses a fixed Otsu threshold inside the known lesion ROI as a stand-in for
    the segmenter, so the drift measured is the compression's fault rather
    than a model's nondeterminism. The quantities mirror the ones
    inference.measure() reports, which is what makes the number meaningful:
    it is the same area and the same mean echo the UI puts in front of a user.

    A 1% area drift on a 4 cm cyst is 0.4 mm of boundary -- inside
    inter-observer variability. A 10% drift is not.
    """
    qualities = qualities or [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    x1, y1, x2, y2 = roi

    def measure(bgr: np.ndarray) -> tuple[float, float]:
        patch = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        area = float((binary > 0).sum())
        echo = float(patch[binary > 0].mean()) if area else 0.0
        return area, echo

    base_area, base_echo = measure(image)
    rows: list[dict[str, object]] = []

    for quality in qualities:
        restored = decode(encode(image, codec, quality))
        area, echo = measure(restored)
        scores = fidelity(image, restored)
        rows.append({
            "codec": codec,
            "quality": quality,
            "ssim": scores["ssim"],
            "psnr_db": scores["psnr_db"],
            "area_px": int(area),
            "area_drift_pct": round(100.0 * (area - base_area) / max(base_area, 1), 2),
            "echo_mean": round(echo, 2),
            "echo_drift": round(echo - base_echo, 2),
        })
    return rows


def equal_size_comparison(image: np.ndarray, target_kb: float = 40.0
                          ) -> list[dict[str, object]]:
    """Compare codecs at the same FILE SIZE rather than the same quality number.

    Quality 80 means different things to JPEG and WebP, so comparing them at
    equal quality compares nothing. Binary-searches each lossy codec for the
    quality that lands nearest `target_kb`, then reports fidelity -- which is
    the comparison that actually answers "which codec should we use".
    """
    rows: list[dict[str, object]] = []

    for codec, spec in CODECS.items():
        if not spec["lossy"]:
            payload = encode(image, codec, 9)
            rows.append({
                "codec": codec, "label": spec["label"], "lossy": False,
                "quality": 9, "kb": round(len(payload) / 1024, 1),
                "note": "lossless -- size is whatever it is",
                **fidelity(image, decode(payload)),
            })
            continue

        lo, hi = 1, 100
        best: dict[str, object] | None = None
        for _ in range(8):
            mid = (lo + hi) // 2
            payload = encode(image, codec, mid)
            kb = len(payload) / 1024
            candidate = {
                "codec": codec, "label": spec["label"], "lossy": True,
                "quality": mid, "kb": round(kb, 1),
                **fidelity(image, decode(payload)),
            }
            if best is None or abs(kb - target_kb) < abs(float(best["kb"]) - target_kb):
                best = candidate
            if kb > target_kb:
                hi = mid - 1
            else:
                lo = mid + 1
        if best:
            best["note"] = f"tuned to ~{target_kb:.0f} KB"
            rows.append(best)

    return rows
