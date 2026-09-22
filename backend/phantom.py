"""Synthetic ultrasound phantom -- a test fixture that needs no dataset.

The MMOTU images are a 1.5 GB download behind a Kaggle account, which makes
them a poor dependency for a smoke test. This module generates a B-mode-like
frame with the properties that actually matter to the rest of the pipeline:

  * multiplicative Rayleigh speckle, not additive Gaussian -- so a filter that
    assumes the wrong noise model fails here the same way it fails on a real
    scan
  * an anechoic (dark) lesion with posterior acoustic enhancement, the bright
    streak below a fluid-filled cyst that is the single most recognisable
    ultrasound artifact
  * a sector field of view with black corners
  * burn-in: caliper crosses, a scale bar and text, which is what
    inference.burnin_mask() is built to remove

It is a phantom, not a simulation. There is no wave physics here -- the
speckle is sampled, not propagated -- so it is good enough to exercise code
and prove a filter changes the right metric, and not good enough to draw any
conclusion about clinical performance.
"""

from __future__ import annotations

import cv2
import numpy as np


def _sector_mask(h: int, w: int) -> np.ndarray:
    """Fan-shaped field of view, as a curvilinear probe produces."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    apex_x, apex_y = w / 2.0, -h * 0.25

    dx, dy = xx - apex_x, yy - apex_y
    radius = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.abs(np.arctan2(dx, dy))

    inside = (angle < np.deg2rad(38)) & (radius < h * 1.18) & (radius > h * 0.28)
    return inside.astype(np.float32)


def generate(width: int = 640, height: int = 480, seed: int = 0,
             lesion: tuple[int, int, int, int] | None = None,
             burn_in: bool = True) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return (bgr_image, lesion_bbox).

    The bbox is the ground-truth lesion extent, which lets enhance.metrics()
    compute a contrast-to-noise ratio against a known region instead of a
    guessed one.
    """
    rng = np.random.default_rng(seed)

    # --- tissue: smooth background echogenicity --------------------------
    base = rng.normal(90, 12, (height, width)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 18)
    base = cv2.normalize(base, None, 55, 135, cv2.NORM_MINMAX)

    # A few brighter fascial planes, so there is real structure to preserve.
    for _ in range(4):
        y = int(rng.integers(int(height * 0.25), int(height * 0.95)))
        thickness = int(rng.integers(2, 5))
        cv2.line(base, (0, y), (width, y + int(rng.integers(-40, 40))),
                 float(rng.uniform(150, 185)), thickness)

    # --- lesion: anechoic ellipse ----------------------------------------
    if lesion is None:
        cx = int(width * rng.uniform(0.38, 0.62))
        cy = int(height * rng.uniform(0.45, 0.62))
        ax = int(width * rng.uniform(0.11, 0.17))
        ay = int(height * rng.uniform(0.13, 0.19))
    else:
        x1, y1, x2, y2 = lesion
        cx, cy, ax, ay = (x1 + x2) // 2, (y1 + y2) // 2, (x2 - x1) // 2, (y2 - y1) // 2

    lesion_mask = np.zeros((height, width), np.float32)
    cv2.ellipse(lesion_mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    lesion_mask = cv2.GaussianBlur(lesion_mask, (0, 0), 3)

    # Fluid is anechoic: drop the interior hard, keep a faint wall.
    base = base * (1.0 - 0.82 * lesion_mask)
    wall = cv2.morphologyEx(
        (lesion_mask > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)
    ).astype(np.float32)
    base += cv2.GaussianBlur(wall, (0, 0), 1.5) * 70

    # Posterior acoustic enhancement: the band below a cyst is brighter,
    # because less energy was attenuated on the way through the fluid.
    posterior = np.zeros((height, width), np.float32)
    cv2.ellipse(posterior, (cx, cy + int(ay * 1.9)), (int(ax * 0.85), int(ay * 0.9)),
                0, 0, 360, 1.0, -1)
    base += cv2.GaussianBlur(posterior, (0, 0), 14) * 34

    # --- speckle: MULTIPLICATIVE Rayleigh ---------------------------------
    # The whole point of the fixture. Rayleigh is the envelope distribution of
    # fully developed speckle under a circular-Gaussian scattering model.
    speckle = rng.rayleigh(scale=1.0, size=(height, width)).astype(np.float32)
    speckle /= speckle.mean()
    speckle = cv2.GaussianBlur(speckle, (0, 0), 0.7)   # finite resolution cell
    image = base * speckle

    # Depth-dependent attenuation.
    depth = np.linspace(1.0, 0.62, height, dtype=np.float32)[:, None]
    image *= depth

    image = np.clip(image, 0, 255)
    image *= _sector_mask(height, width)
    bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # --- burn-in: what inference.burnin_mask() has to remove --------------
    if burn_in:
        # Calipers straddling the lesion, the artifact that makes a model
        # learn "lesion = wherever the calipers are".
        for px, py in ((cx - ax, cy - ay), (cx + ax, cy + ay)):
            cv2.line(bgr, (px - 7, py), (px + 7, py), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(bgr, (px, py - 7), (px, py + 7), (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(bgr, "OVA-PHANTOM  C5-2  MI 0.7", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, "Dist 4.21 cm", (10, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        for i in range(6):                              # depth scale ticks
            y = int(height * 0.15) + i * int(height * 0.13)
            cv2.line(bgr, (width - 16, y), (width - 6, y), (255, 255, 255), 1)

    return bgr, (cx - ax, cy - ay, cx + ax, cy + ay)


def pair(width: int = 640, height: int = 480, seed: int = 0,
         shift: tuple[int, int] = (26, -14), rotation: float = 6.0,
         scale: float = 1.06) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A baseline frame and a 'follow-up' of the same phantom under a known
    similarity transform. Returns (baseline, follow_up, true_matrix_2x3).

    The known matrix is what makes the T5 registration test a measurement:
    the homography SIFT+RANSAC recovers can be compared against the transform
    that was actually applied, instead of being judged by eye.
    """
    baseline, _ = generate(width, height, seed=seed)

    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, rotation, scale)
    matrix[0, 2] += shift[0]
    matrix[1, 2] += shift[1]

    follow_up = cv2.warpAffine(baseline, matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Re-speckle the follow-up: a genuine second acquisition has an
    # independent speckle pattern, and a registration that only works on an
    # identically-speckled copy has proved nothing.
    rng = np.random.default_rng(seed + 991)
    noise = rng.rayleigh(1.0, follow_up.shape[:2]).astype(np.float32)
    noise /= noise.mean()
    noise = cv2.GaussianBlur(noise, (0, 0), 0.7)
    follow_up = np.clip(follow_up.astype(np.float32) * noise[:, :, None], 0, 255).astype(np.uint8)

    return baseline, follow_up, matrix


if __name__ == "__main__":
    image, bbox = generate()
    cv2.imwrite("phantom.png", image)
    baseline, follow_up, matrix = pair()
    cv2.imwrite("phantom_baseline.png", baseline)
    cv2.imwrite("phantom_followup.png", follow_up)
    print(f"phantom.png            lesion bbox {bbox}")
    print(f"phantom_baseline.png / phantom_followup.png written")
    print(f"true transform:\n{matrix}")
