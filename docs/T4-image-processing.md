# T4 — Biomedical image processing

> Syllabus: *"T4: Biomedical Image Processing – enhancement, restoration, segmentation,
> Compression"*

**Code:** [`enhance.py`](../backend/enhance.py), [`inference.py`](../backend/inference.py),
[`compress.py`](../backend/compress.py), [`phantom.py`](../backend/phantom.py) ·
**Pages:** `/imaging`, `/`

Four sub-topics. Restoration and segmentation were already in the project;
enhancement and compression were added.

---

## The fact that governs all four

**Ultrasound speckle is not noise in the usual sense.** It is coherent interference
between echoes from scatterers smaller than the wavelength. Two consequences, and most
generic image processing ignores both:

1. **It is multiplicative**, not additive. The observed image is roughly `f = g · n`,
   not `g + n`. Every filter derived under additive-Gaussian assumptions — Gaussian
   blur, plain non-local means, Wiener — is solving the wrong problem. It over-smooths
   bright regions and barely touches dark ones.
2. **It is deterministic for a fixed geometry.** Averaging repeated frames does not
   remove it. Single-frame filtering is the only option.

A log transform turns the product into a sum:

```
log(g · n) = log g + log n
```

so an additive denoiser applied in the **log domain** is principled. `homomorphic=True`
does this, and the difference shows up in the metrics rather than being a matter of
taste.

---

## 1. Enhancement

### Contrast

**Global histogram equalisation** — the baseline that shows why CLAHE exists. Remaps
intensities so the histogram is flat across the *whole* frame, which on ultrasound
amplifies noise in the anechoic interior of a cyst — the exact region segmentation
needs to stay smooth.

**CLAHE** — equalises within small tiles. `clip_limit` caps each tile histogram before
equalising and redistributes the excess. **That clipping is the noise control**: an
unclipped tile histogram in a flat dark region has enormous slope and turns speckle
into structure. 2.0 over an 8×8 grid is the ultrasound default; higher looks sharper
and hallucinates texture.

**Gamma** — `out = 255·(in/255)^γ`, via a 256-entry LUT rather than per-pixel `pow`.

### Speckle reduction

| Method | Mechanism | Trade-off |
|---|---|---|
| Median | order statistic | edge-tolerant; loses structure thinner than `ksize//2` |
| Bilateral | weights by spatial **and** intensity distance | keeps the cyst wall, smooths the interior; O(d²)/pixel |
| Non-local means | averages pixels with similar *neighbourhoods* | exploits repeated texture; run in log domain |
| Anisotropic diffusion | Perona–Malik: `dI/dt = div(c(|∇I|)·∇I)` | diffuses **along** edges, not across |

Anisotropic diffusion is hand-written rather than taken from `cv2.ximgproc`, which is a
contrib module absent from `opencv-python-headless`. `gamma_step` must stay ≤ 0.25 for
stability on a 4-neighbour stencil; the function raises above that rather than
silently diverging.

### Measured on the phantom

| Method | Sharpness | Speckle index | Entropy | CNR |
|---|---|---|---|---|
| Original | 1412.9 | 0.8779 | 5.346 | 0.283 |
| Histogram equalisation | **9375.7** | **1.0149** | 5.228 | 0.335 |
| CLAHE | 4447.2 | 0.9192 | **6.054** | 0.177 |
| Gamma 0.6 | 1450.5 | 0.7958 | 5.276 | 0.129 |
| Median | 160.0 | 0.8316 | 5.039 | 0.303 |
| Bilateral | 703.1 | 0.8414 | 5.185 | 0.299 |
| **Non-local means (log)** | 483.7 | **0.8287** | 4.478 | **0.320** |
| Anisotropic diffusion | 603.4 | 0.8363 | 5.117 | 0.307 |
| Unsharp mask | 4550.2 | 0.9796 | 5.644 | 0.257 |

**Reading these together is the point, and any single column misleads:**

- Histogram equalisation posts by far the **highest sharpness** and the **worst speckle
  index**. It amplified noise — which is what Laplacian variance measures when there is
  little genuine detail to sharpen. A sharpness-only evaluation would rank it first.
- CLAHE has the highest **entropy**, i.e. retains the most information, because
  clipping stops it flattening the anechoic interior. Its CNR *drops* though: boosting
  local contrast everywhere reduces the global lesion-vs-background ratio. Two
  legitimate metrics disagreeing about the same operation.
- The log-domain NLM gets the lowest speckle index **and** the best CNR among the
  speckle filters — the homomorphic treatment earning its place.
- The unsharp mask raises speckle to 0.98, confirming it belongs **after** a speckle
  filter, never before.

### Metrics

Reference-free, because a real scan has no clean ground truth (PSNR/SSIM need one and
live in `compress.py`, where the original *is* available).

- **sharpness** — variance of the Laplacian. Rises with detail *and* with noise.
- **speckle_index** — σ/μ. The standard ultrasound speckle measure.
- **entropy** — Shannon entropy of the histogram, in bits.
- **cnr** — contrast-to-noise between an ROI and the rest. The one that tracks whether
  a lesion actually became easier to see; needs a known ROI, which the phantom supplies.

---

## 2. Restoration

Already implemented in [`inference.py`](../backend/inference.py) and unchanged.

**Burn-in removal.** Sonographer calipers sit *inside* the lesion. Left in place, the
model learns "lesion = wherever the calipers are" and fails on unmarked images. The
pipeline:

1. Detect high-saturation or near-white pixels: `hsv[:,:,1] > 60 | gray > 235`
2. Keep only small, compact connected components — `3 < area < 1200`,
   `compactness > 0.15`. Real anatomy is neither, and the compactness test rejects long
   sparse streaks.
3. Dilate 5×5 to cover antialiased edges.
4. Telea inpainting.

This runs **before inference**, not just for display — the model never sees the
calipers.

---

## 3. Segmentation

Already implemented. YOLOv11n-seg instance masks, plus classical contour analysis in
`measure()`: external contours, convex hull, max caliper as the largest pairwise
distance across hull vertices, circularity `4πA/P²`, solidity `A/A_hull`, and echo
statistics inside the mask.

All measurements are in **pixels**. Conversion to mm needs the scanner scale bar or
DICOM `PixelSpacing`, neither of which this prototype reads — stated in four places in
the codebase because it is the single most likely thing for a reader to assume wrongly.

See [T5](T5-sift-ransac-cnn.md) for the CNN itself.

---

## 4. Compression

### The question

Not "how small can it get" but "how small before it stops being diagnostic" — and those
have different answers.

**Lossless** (PNG; and the lossless JPEG-2000 / JPEG-LS modes DICOM specifies) gets
only ~4× here:

| PNG level | Size | Ratio |
|---|---|---|
| 0 | 902.0 KB | 1.00× |
| 6 | 226.2 KB | 3.98× |
| 9 | 211.0 KB | 4.27× |

That poor ratio **is** a result: speckle is high-entropy and looks like noise to an
entropy coder, which is exactly what it is.

**Lossy** reaches 70× — by discarding high-frequency detail, which on ultrasound *is*
the speckle. Compressed frames often look *cleaner*.

### Comparing codecs correctly

"Quality 80" means different things to JPEG and WebP, so comparing at equal quality
compares nothing. `equal_size_comparison` binary-searches each codec to the same file
size:

| Codec | Quality | Size | PSNR | SSIM |
|---|---|---|---|---|
| PNG | 9 | 211.0 KB | ∞ (lossless) | 1.00000 |
| JPEG | 59 | 39.8 KB | 32.52 dB | 0.93164 |
| **WebP** | 72 | 40.3 KB | **35.21 dB** | **0.95977** |

WebP wins by ~2.7 dB at the same size, as expected from its better rate-distortion
behaviour.

### Finding — SSIM is not monotonic in quality

JPEG on the phantom:

| Quality | PSNR (dB) | SSIM |
|---|---|---|
| 95 | 44.17 | 0.99343 |
| 90 | 39.62 | 0.98277 |
| 80 | 35.63 | 0.96182 |
| **70** | 33.71 | **0.90590** |
| **60** | 32.62 | **0.93296** |
| 50 | 31.83 | 0.92118 |
| **40** | 31.05 | **0.86858** |
| **30** | 30.17 | **0.88957** |
| 10 | 26.79 | 0.76444 |

PSNR falls monotonically. **SSIM does not** — it is higher at q=60 than q=70, and
higher at q=30 than q=40.

This is not a bug. SSIM's structure term compares local variance patterns, and in a
speckle-dominated image local variance is itself close to random — so whether a given
quantisation table happens to preserve one speckle realisation is largely luck. It is
direct evidence that SSIM, designed for natural images, has shaky foundations on
ultrasound. Quote PSNR beside it, and prefer a task metric to either.

### The metric that matters — measurement drift

Fidelity is easy to quote; what matters is whether the *answer* changes.
`measurement_drift` re-measures the lesion after each compression, using a fixed Otsu
threshold inside the known ROI as a deterministic stand-in for the segmenter (so the
drift measured is compression's fault, not model nondeterminism).

| Quality | SSIM | Area drift | Echo drift |
|---|---|---|---|
| 95 | 0.9934 | +0.07% | +0.02 |
| 80 | 0.9618 | −0.02% | −0.01 |
| 60 | 0.9330 | −0.54% | −0.23 |
| 30 | 0.8896 | −0.61% | −0.33 |
| 20 | 0.8552 | −0.70% | −0.59 |

**Fidelity degrades far faster than the measurement does.** SSIM falls to 0.855 while
the lesion area moves under a percent.

**This does not generalise, and the code says so.** The phantom lesion is a large,
high-contrast, sharp-walled anechoic region — the easiest possible segmentation target.
A small isoechoic lesion with a diffuse border sits much closer to the detail JPEG
discards first and should be expected to drift considerably more. Re-run
`measurement_drift()` on real images before concluding anything from these numbers.

### Regulatory note

Lossy compression of diagnostic images is restricted in most jurisdictions; where
permitted, the ratio is capped by modality and must be disclosed. Nothing here is a
recommendation to use it clinically.

---

## The phantom

[`phantom.py`](../backend/phantom.py) generates a B-mode-like frame so none of the above
needs the 1.5 GB MMOTU download. It models what matters to this code:

- **multiplicative Rayleigh speckle** — so a filter assuming the wrong noise model fails
  here the way it fails on a real scan
- an **anechoic lesion with posterior acoustic enhancement** — the bright band below a
  fluid-filled cyst, the most recognisable ultrasound artifact
- a **sector field of view** with black corners
- **burn-in**: caliper crosses, scale ticks and text, for `burnin_mask()` to remove

It is a phantom, not a simulation. There is no wave physics — the speckle is sampled,
not propagated. Good enough to exercise code and prove a filter moves the right metric;
**not** good enough to support any claim about clinical performance.

---

Previous: [T3](T3-redundant-and-missing-data.md) · Next: [T5](T5-sift-ransac-cnn.md)
