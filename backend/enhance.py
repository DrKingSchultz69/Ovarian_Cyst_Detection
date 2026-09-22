"""T4 part one -- image enhancement and restoration.

Ultrasound is the hardest common modality to enhance, for one reason:

    SPECKLE IS NOT NOISE IN THE USUAL SENSE.

    It is coherent interference between echoes from scatterers smaller than
    the wavelength. Two consequences follow, and most generic filters ignore
    both:

      1. It is MULTIPLICATIVE, not additive. The observed image is roughly
         f = g * n, not g + n. Every filter derived under an additive-Gaussian
         assumption -- Gaussian blur, plain non-local means, Wiener -- is
         solving the wrong problem, and removes signal along with speckle in
         bright regions while barely touching dark ones.

      2. It is deterministic for a fixed geometry. Averaging repeated frames
         does not remove it, which is why single-frame filtering is the only
         option here.

    A log transform turns the product into a sum, so an additive denoiser
    applied in the log domain IS principled. `homomorphic=True` does that, and
    the difference is visible in the metrics rather than a matter of taste.

WHY ENHANCEMENT MATTERS HERE
    inference.py already restores the image (burn-in inpainting) before the
    model sees it. This module is the step before that: it improves what the
    inpainting and the segmentation have to work with. Everything is
    measurable -- see metrics() -- so a claim that a filter helped can be
    checked instead of asserted.

    No web-framework imports, same rule as inference.py.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _match_shape(result: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Return a single-channel result as 3-channel when the input was colour,
    so a filter can be dropped into a BGR pipeline without special-casing."""
    if like.ndim == 3 and result.ndim == 2:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def _homomorphic(fn: Callable[[np.ndarray], np.ndarray], gray: np.ndarray) -> np.ndarray:
    """Apply an additive-noise filter in the log domain.

    log(g * n) = log(g) + log(n), so multiplicative speckle becomes additive
    and the filter's assumptions hold. The +1 avoids log(0); the rescale back
    to 0-255 is what makes the result comparable to the other filters.
    """
    work = np.log1p(gray.astype(np.float32))
    work = cv2.normalize(work, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filtered = fn(work).astype(np.float32)
    restored = np.expm1(filtered / 255.0 * np.log1p(255.0))
    return cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------


def histogram_equalisation(image: np.ndarray) -> np.ndarray:
    """Global histogram equalisation.

    Included as the baseline that shows why CLAHE exists. It remaps intensities
    so the histogram is flat across the WHOLE frame, which on ultrasound
    amplifies noise in the anechoic (dark) interior of a cyst -- exactly the
    region the segmentation needs to stay smooth.
    """
    return _match_shape(cv2.equalizeHist(_as_gray(image)), image)


def clahe(image: np.ndarray, clip_limit: float = 2.0,
          tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    Equalises within small tiles, so local contrast improves without the
    global stretch. `clip_limit` caps the histogram before equalising and
    redistributes the excess -- that clipping IS the noise control, because an
    unclipped tile histogram in a flat dark region has enormous slope and
    turns speckle into structure.

    The standard default for ultrasound is a 2.0 clip over an 8x8 grid.
    Higher clip limits look sharper and hallucinate texture.
    """
    operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return _match_shape(operator.apply(_as_gray(image)), image)


def gamma(image: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Power-law intensity mapping, out = 255 * (in/255) ** gamma.

    gamma < 1 brightens mid-tones and reveals low-echo internal structure;
    gamma > 1 darkens them and suppresses background haze. A lookup table is
    used rather than per-pixel maths -- 256 pow() calls instead of millions.
    """
    table = np.array([((i / 255.0) ** value) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(image, table)


# ---------------------------------------------------------------------------
# Speckle reduction
# ---------------------------------------------------------------------------


def median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter. The cheap baseline for impulse-like speckle.

    Preserves edges better than a mean filter because the median of a window
    straddling an edge is still a value from one side of it, never an average
    of both. Loses thin structures narrower than ksize//2.
    """
    return cv2.medianBlur(image, ksize | 1)   # ksize must be odd


def bilateral(image: np.ndarray, d: int = 9, sigma_colour: float = 75.0,
              sigma_space: float = 75.0) -> np.ndarray:
    """Edge-preserving smoothing.

    Weights neighbours by BOTH spatial distance and intensity difference, so
    pixels across a boundary contribute almost nothing. The cyst wall survives
    while the interior smooths -- which is the behaviour segmentation wants.
    Slow: O(d^2) per pixel with no separable form.
    """
    return cv2.bilateralFilter(image, d, sigma_colour, sigma_space)


def non_local_means(image: np.ndarray, h: float = 10.0,
                    homomorphic: bool = True) -> np.ndarray:
    """Non-local means: average pixels with similar NEIGHBOURHOODS, wherever
    they are in the frame, rather than pixels that are merely nearby.

    Strong on textured tissue because ultrasound repeats its texture. Applied
    in the log domain by default -- NLM assumes additive noise, and speckle is
    multiplicative, so the untransformed version over-smooths bright regions.
    Set homomorphic=False to see that failure.
    """
    gray = _as_gray(image)
    fn = lambda g: cv2.fastNlMeansDenoising(g, None, h, 7, 21)  # noqa: E731
    result = _homomorphic(fn, gray) if homomorphic else fn(gray)
    return _match_shape(result, image)


def anisotropic_diffusion(image: np.ndarray, iterations: int = 15,
                          kappa: float = 30.0, gamma_step: float = 0.15) -> np.ndarray:
    """Perona-Malik diffusion: blur along edges, never across them.

    Iteratively solves dI/dt = div(c(|grad I|) * grad I) where the conductance
    c falls off as the gradient rises, so smoothing stops at boundaries. Hand
    written rather than taken from cv2.ximgproc, which is a contrib module and
    absent from opencv-python-headless.

      kappa       gradient above which an edge is preserved. Too high and this
                  degenerates to Gaussian blur.
      gamma_step  step size. Must stay <= 0.25 for stability on a 4-neighbour
                  stencil; larger values oscillate and then diverge.
    """
    if gamma_step > 0.25:
        raise ValueError("gamma_step > 0.25 is unstable on a 4-neighbour stencil")

    out = _as_gray(image).astype(np.float32)

    for _ in range(iterations):
        # Forward differences in four directions.
        dn = np.roll(out, -1, axis=0) - out
        ds = np.roll(out, 1, axis=0) - out
        de = np.roll(out, -1, axis=1) - out
        dw = np.roll(out, 1, axis=1) - out

        # Perona-Malik conductance #2: favours wide regions over small ones.
        cn = np.exp(-(dn / kappa) ** 2)
        cs = np.exp(-(ds / kappa) ** 2)
        ce = np.exp(-(de / kappa) ** 2)
        cw = np.exp(-(dw / kappa) ** 2)

        out = out + gamma_step * (cn * dn + cs * ds + ce * de + cw * dw)

    return _match_shape(np.clip(out, 0, 255).astype(np.uint8), image)


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------


def unsharp_mask(image: np.ndarray, sigma: float = 1.5,
                 amount: float = 1.0) -> np.ndarray:
    """out = image + amount * (image - blur(image)).

    The difference term is the high-frequency content, so adding it back
    boosts edges. On ultrasound this amplifies speckle just as readily as
    anatomy, so it belongs AFTER a speckle filter, never before -- the
    pipeline order is the whole trick.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


# ---------------------------------------------------------------------------
# Quality metrics -- the reason any of this is checkable
# ---------------------------------------------------------------------------


def metrics(image: np.ndarray, roi: tuple[int, int, int, int] | None = None
            ) -> dict[str, float]:
    """Reference-free image quality measures.

    Reference-free because there is no clean ground truth for a real scan --
    PSNR and SSIM need one, so they belong in compress.py where the original
    IS available, not here.

      sharpness  variance of the Laplacian. Higher is crisper; it also rises
                 with noise, so it must be read next to speckle_index.
      speckle_index  sigma/mu over the image. THE standard ultrasound speckle
                 measure: lower means less speckle FOR THE SAME BRIGHTNESS --
                 and that qualifier is load-bearing.

                 CONFOUND, measured on the phantom: a pure intensity remap
                 changes it without touching a single noise property.

                     original            mean 52.9  sd 46.0  SI 0.8711
                     gamma 0.6 (bright)  mean 81.4  sd 64.9  SI 0.7971
                     gamma 1.6 (dark)    mean 28.2  sd 28.8  SI 1.0203
                     median (denoise)    mean 51.6  sd 42.8  SI 0.8293

                 Gamma 0.6 posts a BETTER speckle index than the median
                 filter, purely by raising the mean. It denoised nothing.
                 So SI only compares filters that leave brightness alone --
                 i.e. within the `speckle` group -- and a contrast operator
                 must never be ranked against a denoiser on it. compare()
                 returns `group` for exactly this reason.
      entropy    Shannon entropy of the histogram, in bits. Information
                 content; collapses when a filter flattens the image.
      cnr        contrast-to-noise between `roi` and everything else. Only
                 computed when an ROI is given -- it is the metric that
                 actually tracks whether a lesion became easier to see.
    """
    gray = _as_gray(image).astype(np.float64)

    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    mean = float(gray.mean())
    std = float(gray.std())

    histogram = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    probability = histogram / max(histogram.sum(), 1.0)
    nonzero = probability[probability > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    out = {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "sharpness": round(float(laplacian.var()), 2),
        "speckle_index": round(std / mean, 4) if mean else 0.0,
        "entropy": round(entropy, 3),
    }

    if roi is not None:
        x1, y1, x2, y2 = roi
        mask = np.zeros(gray.shape, bool)
        mask[y1:y2, x1:x2] = True
        inside, outside = gray[mask], gray[~mask]
        if inside.size and outside.size:
            denominator = np.sqrt(inside.var() + outside.var())
            out["cnr"] = round(
                float(abs(inside.mean() - outside.mean()) / denominator), 3
            ) if denominator else 0.0

    return out


# ---------------------------------------------------------------------------
# Registry -- what the API exposes
# ---------------------------------------------------------------------------

METHODS: dict[str, dict[str, object]] = {
    "histogram_equalisation": {
        "fn": histogram_equalisation, "group": "contrast",
        "label": "Global histogram equalisation",
        "note": "Baseline. Amplifies noise in anechoic regions.",
    },
    "clahe": {
        "fn": clahe, "group": "contrast", "label": "CLAHE",
        "note": "Local contrast with a clip limit. The ultrasound default.",
    },
    "gamma": {
        "fn": gamma, "group": "contrast", "label": "Gamma correction (0.6)",
        "note": "Power-law remap of mid-tones.",
        # gamma()'s own default is 1.0, which is the identity -- correct as a
        # signature, useless in a comparison strip. compare() uses this.
        "demo": {"value": 0.6},
    },
    "median": {
        "fn": median, "group": "speckle", "label": "Median filter",
        "note": "Cheap, edge-tolerant, loses thin structure.",
    },
    "bilateral": {
        "fn": bilateral, "group": "speckle", "label": "Bilateral filter",
        "note": "Edge-preserving. Smooths the cyst interior, keeps the wall.",
    },
    "non_local_means": {
        "fn": non_local_means, "group": "speckle", "label": "Non-local means (log domain)",
        "note": "Exploits repeated texture. Homomorphic by default.",
    },
    "anisotropic_diffusion": {
        "fn": anisotropic_diffusion, "group": "speckle", "label": "Anisotropic diffusion",
        "note": "Perona-Malik. Diffuses along edges, not across them.",
    },
    "unsharp_mask": {
        "fn": unsharp_mask, "group": "sharpen", "label": "Unsharp mask",
        "note": "Run AFTER speckle reduction or it amplifies the speckle.",
    },
}


def apply(name: str, image: np.ndarray, **kwargs) -> np.ndarray:
    """Apply one registered method by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method {name!r}. Known: {', '.join(METHODS)}")
    return METHODS[name]["fn"](image, **kwargs)  # type: ignore[operator]


def compare(image: np.ndarray, names: list[str] | None = None,
            roi: tuple[int, int, int, int] | None = None
            ) -> list[dict[str, object]]:
    """Run several methods and measure each. Returns the original first, so a
    caller can render a strip and read the metrics underneath."""
    names = names or list(METHODS)
    rows: list[dict[str, object]] = [
        {"name": "original", "label": "Original", "group": "-",
         "image": image, "metrics": metrics(image, roi)}
    ]
    for name in names:
        result = apply(name, image, **METHODS[name].get("demo", {}))  # type: ignore[arg-type]
        rows.append({
            "name": name,
            "label": METHODS[name]["label"],
            "group": METHODS[name]["group"],
            "note": METHODS[name]["note"],
            "image": result,
            "metrics": metrics(result, roi),
        })
    return rows
