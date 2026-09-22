"""T5 -- SIFT, RANSAC and image registration.

The clinical question this answers: a patient returns in three months, and the
cyst must be compared against the baseline scan. The probe was at a different
angle, at a different depth setting, on a different day. Before any
measurement can be compared, the two frames have to be put in the same
coordinate system. That is registration, and the classical pipeline is:

    SIFT  detect and describe distinctive points, independently in each image
    match nearest-neighbour descriptors, filtered by Lowe's ratio test
    RANSAC fit one geometric transform, rejecting the matches that disagree
    warp  resample the follow-up onto the baseline grid

WHY THE CNN DOES NOT REPLACE THIS
    The segmentation model (T5's third technique, already in inference.py)
    finds WHAT is in one frame. It says nothing about the geometric relation
    between two frames. These are complementary, not competing: register
    first, then compare the per-frame segmentations in a shared frame.

WHY SIFT IS HARD ON ULTRASOUND -- and this is the honest headline
    SIFT keypoints are extrema of a difference-of-Gaussian scale space, and
    speckle produces thousands of them. They are repeatable only if the
    speckle pattern is repeatable, and it is NOT: it depends on the exact
    probe geometry, so a genuine second acquisition has an independent speckle
    field. The keypoints that survive between real acquisitions are the ones
    on true anatomical structure, and they are a minority.

    The practical consequence is that the ratio test must be strict and
    RANSAC must expect a high outlier fraction. phantom.pair() re-speckles the
    follow-up precisely so this module is tested against that difficulty
    rather than against a conveniently identical copy.

PATENT NOTE
    SIFT's patent expired in March 2020 and it is in the main OpenCV module
    from 4.4 onward -- cv2.SIFT_create() needs no contrib build.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

# Lowe's ratio. A match is kept only when the best descriptor distance is
# clearly better than the second best; 0.7-0.8 is the usual band. Tighter than
# the 0.75 default here because ultrasound speckle generates many near-ties.
RATIO = 0.72

# RANSAC reprojection tolerance, in pixels. Too tight and a correct model is
# rejected for sub-pixel localisation error; too loose and a wrong model finds
# enough accidental support to look good.
RANSAC_REPROJ = 3.0


@dataclasses.dataclass
class RegistrationResult:
    ok: bool
    model: str
    matrix: np.ndarray | None
    n_keypoints: tuple[int, int]
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    rmse_px: float | None
    warped: np.ndarray | None
    match_visualisation: np.ndarray | None
    error: str | None = None

    def as_dict(self) -> dict:
        """JSON-safe summary, without the images."""
        return {
            "ok": self.ok,
            "model": self.model,
            "matrix": None if self.matrix is None else self.matrix.round(5).tolist(),
            "keypoints_baseline": self.n_keypoints[0],
            "keypoints_followup": self.n_keypoints[1],
            "matches_after_ratio_test": self.n_matches,
            "inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "rmse_px": None if self.rmse_px is None else round(self.rmse_px, 3),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Detection and description
# ---------------------------------------------------------------------------


def detect(image: np.ndarray, n_features: int = 0, contrast_threshold: float = 0.04
           ) -> tuple[list, np.ndarray | None]:
    """SIFT keypoints and 128-dimensional descriptors.

    contrast_threshold filters weak extrema. RAISING it is the main defence
    against speckle keypoints: the default 0.04 is tuned for photographs, and
    on ultrasound it returns thousands of points sitting on noise. The cost of
    raising it is losing genuine low-contrast anatomy, so it is exposed rather
    than hard-coded.

    Each descriptor is a 4x4 grid of 8-bin gradient-orientation histograms.
    That construction is what makes SIFT invariant to rotation (orientations
    are measured relative to a dominant direction) and to scale (the patch is
    sampled at the keypoint's own scale), and robust to illumination (the
    vector is normalised).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return list(keypoints), descriptors


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = RATIO) -> list:
    """k-NN match with Lowe's ratio test.

    For each descriptor in A, take its two nearest neighbours in B. If the
    closest is not much closer than the runner-up, the match is ambiguous and
    is discarded. On repetitive texture -- which speckle is, in the extreme --
    this is what prevents a flood of confident nonsense.

    L2 is the correct metric for SIFT descriptors (they are gradient
    histograms, not binary strings; Hamming would be for ORB/BRIEF).
    """
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)

    kept = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            kept.append(best)
    return kept


# ---------------------------------------------------------------------------
# Robust model fitting
# ---------------------------------------------------------------------------


def fit(src_pts: np.ndarray, dst_pts: np.ndarray, model: str = "affine",
        reproj: float = RANSAC_REPROJ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """RANSAC fit of a geometric transform.

    RANSAC, in one paragraph: repeatedly draw the minimum number of
    correspondences needed to define the model (3 for affine, 4 for
    homography), fit it, count how many of the remaining matches it explains
    within `reproj` pixels, and keep the model with the most support. Because
    it only ever fits to a minimal clean-by-luck subset, a majority of wrong
    matches does not drag the estimate -- which least squares cannot survive.

    MODEL CHOICE MATTERS MORE THAN IT LOOKS
      affine       6 DOF. Rotation, scale, shear, translation. Straight lines
                   stay straight and parallel lines stay parallel.
      homography   8 DOF. Adds perspective. Correct for a planar scene viewed
                   from different angles -- and WRONG here. Two ultrasound
                   sweeps are different cross-sections of a deforming 3D
                   organ, not two views of a plane. The extra freedom lets a
                   homography fold the image to fit outliers, which looks
                   like a better inlier count and is a worse answer.

    MEASURED ON phantom.pair(), where the true transform is known:

        model        inliers   RMSE px   true corner error px
        rigid          49/62     0.933                  0.421
        affine         49/62     0.947                  0.695
        homography     48/62     0.839                  2.136

    The homography has the BEST reprojection RMSE and the WORST actual error,
    by a factor of five. That is overfitting made visible: eight degrees of
    freedom absorb noise the correct model cannot, so the residual it reports
    on its own inliers gets smaller while the transform gets further from the
    truth. It is also the reason RMSE alone must never be used to choose a
    model -- only a held-out ground truth separates them, and in the clinic
    there is not one.

    Affine is therefore the default. Rigid is more conservative still and
    scores best here, which is the expected ordering: the phantom transform
    IS a similarity (rotation + uniform scale + shift), so the model that
    matches the true generating process wins.
    """
    if len(src_pts) < 4:
        return None, None

    if model == "homography":
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj)
    elif model == "affine":
        matrix, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=reproj
        )
    elif model == "rigid":
        matrix, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=reproj
        )
    else:
        raise ValueError(f"Unknown model {model!r}")

    return matrix, mask


def _reprojection_rmse(src: np.ndarray, dst: np.ndarray,
                       matrix: np.ndarray, mask: np.ndarray) -> float:
    """RMSE over the inliers only.

    Over all matches it would just measure how many outliers there were, which
    is the inlier ratio in disguise. Over the inliers it measures how well the
    model actually fits the correspondences it claims to explain.
    """
    inliers = mask.ravel().astype(bool)
    if inliers.sum() == 0:
        return float("nan")

    source = src[inliers].reshape(-1, 1, 2)
    target = dst[inliers].reshape(-1, 2)

    if matrix.shape == (3, 3):
        projected = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    else:
        projected = cv2.transform(source, matrix).reshape(-1, 2)

    return float(np.sqrt(((projected - target) ** 2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def register(baseline: np.ndarray, follow_up: np.ndarray, model: str = "affine",
             ratio: float = RATIO, contrast_threshold: float = 0.04,
             draw: bool = True) -> RegistrationResult:
    """Align `follow_up` onto `baseline`.

    Returns a result object rather than raising, matching the convention in
    inference.analyse_bytes(): a registration that cannot find enough
    structure is a normal outcome on ultrasound, not an exception.
    """
    kp_a, desc_a = detect(baseline, contrast_threshold=contrast_threshold)
    kp_b, desc_b = detect(follow_up, contrast_threshold=contrast_threshold)

    def failure(message: str, n_matches: int = 0) -> RegistrationResult:
        return RegistrationResult(
            ok=False, model=model, matrix=None,
            n_keypoints=(len(kp_a), len(kp_b)), n_matches=n_matches,
            n_inliers=0, inlier_ratio=0.0, rmse_px=None,
            warped=None, match_visualisation=None, error=message,
        )

    matches = match(desc_a, desc_b, ratio)
    if len(matches) < 4:
        return failure(
            f"Only {len(matches)} matches survived the ratio test; "
            f"4 are needed. Try a lower contrast_threshold or a looser ratio.",
            len(matches),
        )

    # queryIdx indexes the FOLLOW-UP when we ask for the transform that maps
    # follow-up coordinates onto baseline coordinates, so order the point sets
    # accordingly: src = follow-up, dst = baseline.
    src = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = fit(src, dst, model)
    if matrix is None or mask is None:
        return failure("RANSAC found no consistent model.", len(matches))

    n_inliers = int(mask.sum())
    inlier_ratio = n_inliers / len(matches)
    rmse = _reprojection_rmse(src.reshape(-1, 2), dst.reshape(-1, 2), matrix, mask)

    height, width = baseline.shape[:2]
    if matrix.shape == (3, 3):
        warped = cv2.warpPerspective(follow_up, matrix, (width, height))
    else:
        warped = cv2.warpAffine(follow_up, matrix, (width, height))

    visualisation = None
    if draw:
        visualisation = cv2.drawMatches(
            baseline, kp_a, follow_up, kp_b, matches, None,
            matchesMask=mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    return RegistrationResult(
        ok=True, model=model, matrix=matrix,
        n_keypoints=(len(kp_a), len(kp_b)), n_matches=len(matches),
        n_inliers=n_inliers, inlier_ratio=inlier_ratio, rmse_px=rmse,
        warped=warped, match_visualisation=visualisation,
    )


# ---------------------------------------------------------------------------
# Visual comparison
# ---------------------------------------------------------------------------


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 48) -> np.ndarray:
    """Interleave two aligned images in a checkerboard.

    The standard registration QA view: misalignment shows up as structure
    breaking at the tile boundaries, which the eye catches instantly and a
    difference image does not.
    """
    out = a.copy()
    height, width = a.shape[:2]
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if ((x // tile) + (y // tile)) % 2:
                out[y:y + tile, x:x + tile] = b[y:y + tile, x:x + tile]
    return out


def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference, colour-mapped. Bright means poorly aligned."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    return cv2.applyColorMap(cv2.absdiff(ga, gb), cv2.COLORMAP_INFERNO)


def compare_models(baseline: np.ndarray, follow_up: np.ndarray,
                   truth: np.ndarray | None = None) -> list[dict]:
    """Run every model and report the fit.

    When `truth` (a 2x3 affine) is supplied, the recovered transform is scored
    against it by how far it moves the four image corners -- a geometric error
    in pixels, which is interpretable, rather than a matrix norm, which is not.
    """
    rows = []
    height, width = baseline.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)

    for model in ("rigid", "affine", "homography"):
        result = register(baseline, follow_up, model=model, draw=False)
        row = result.as_dict()

        if truth is not None and result.ok and result.matrix is not None:
            true_corners = cv2.transform(corners, truth).reshape(-1, 2)
            if result.matrix.shape == (3, 3):
                got = cv2.perspectiveTransform(corners, result.matrix).reshape(-1, 2)
            else:
                got = cv2.transform(corners, result.matrix).reshape(-1, 2)
            # The recovered transform maps follow-up -> baseline; the truth
            # maps baseline -> follow-up. Compare against the truth inverse.
            inverse = cv2.invertAffineTransform(truth)
            expected = cv2.transform(corners, inverse).reshape(-1, 2)
            row["corner_error_px"] = round(
                float(np.sqrt(((got - expected) ** 2).sum(axis=1)).mean()), 3
            )
            del true_corners
        rows.append(row)

    return rows
