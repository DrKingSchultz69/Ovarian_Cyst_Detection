# T5 — SIFT, RANSAC, CNN

> Syllabus: *"T5: Biomedical Image Analysis Techniques – SIFT, RANSAC, CNN"*

**Code:** [`registration.py`](../backend/registration.py),
[`inference.py`](../backend/inference.py) · **Pages:** `/registration`, `/`

Three techniques that are complementary, not competing — which is presumably why the
syllabus groups them.

---

## The clinical problem

A patient returns in three months and the cyst must be compared against baseline. The
probe is at a different angle, at a different depth setting, on a different day. Before
any measurement can be compared, the two frames need a shared coordinate system.

That is **registration**, and the classical pipeline is exactly SIFT → matching →
RANSAC.

The CNN answers a different question — *what is in this frame* — and says nothing about
the geometry between two. Register first, then compare the per-frame segmentations in a
shared frame.

---

# SIFT

## What it computes

Scale-Invariant Feature Transform. Two stages:

**Detection** — find extrema of a difference-of-Gaussian scale space. A DoG approximates
the scale-normalised Laplacian, so extrema are blob-like structures located in both
space *and* scale.

**Description** — around each keypoint, build a 4×4 grid of 8-bin gradient-orientation
histograms: a 128-dimensional vector. That construction is what buys the invariances:

| Invariance | How |
|---|---|
| Rotation | orientations measured relative to the keypoint's dominant direction |
| Scale | patch sampled at the keypoint's own detected scale |
| Illumination | vector normalised, then clipped and renormalised |

**Patent note:** SIFT's patent expired in March 2020. It is in main OpenCV from 4.4 —
`cv2.SIFT_create()` needs no contrib build.

## Why it is hard on ultrasound — the honest headline

SIFT keypoints are DoG extrema, and **speckle produces thousands of them**. They are
repeatable only if the speckle pattern is repeatable, and it is **not**: speckle depends
on exact probe geometry, so a genuine second acquisition has an independent speckle
field.

Measured on the phantom pair:

```
keypoints in baseline     3078
keypoints in follow-up    4479
matches after ratio test    62      ← ~2% survive
RANSAC inliers              49
```

Roughly 2% of keypoints produce a usable correspondence. The survivors are the ones
sitting on true anatomical structure, and they are a small minority.

`phantom.pair()` **re-speckles the follow-up independently** precisely so this
difficulty is real. A registration demo that aligns an image to a transformed copy of
itself — same speckle — has proved nothing, because the speckle is then the strongest
and most repeatable signal in the frame.

`contrast_threshold` is exposed rather than hard-coded for this reason. The OpenCV
default of 0.04 is tuned for photographs; raising it is the main defence against
speckle keypoints, at the cost of losing genuine low-contrast anatomy.

---

# Matching — Lowe's ratio test

For each descriptor in A take its two nearest neighbours in B. Keep the match only if:

```
d(best) < ratio × d(second_best)
```

If the closest is not *clearly* closer than the runner-up, the match is ambiguous.

On repetitive texture — and speckle is repetitive in the extreme — this is what
prevents a flood of confident nonsense. The threshold here is **0.72**, tighter than
Lowe's usual 0.75–0.8, because ultrasound generates so many near-ties.

L2 is the correct metric for SIFT descriptors: they are gradient histograms, not binary
strings. Hamming would be for ORB/BRIEF.

---

# RANSAC

## The algorithm in one paragraph

Repeatedly draw the minimum number of correspondences needed to define the model (3 for
affine, 4 for homography), fit it, count how many remaining matches it explains within
`reproj` pixels, and keep the model with most support. Because it only ever fits to a
minimal *clean-by-luck* subset, a majority of wrong matches does not drag the estimate —
which ordinary least squares cannot survive.

`RANSAC_REPROJ = 3.0` px. Too tight and a correct model is rejected over sub-pixel
localisation error; too loose and a wrong model finds enough accidental support to look
convincing.

## Model choice matters more than it looks

| Model | DOF | Preserves |
|---|---|---|
| Rigid / similarity | 4 | rotation, uniform scale, translation |
| **Affine** | 6 | straight lines, parallelism |
| Homography | 8 | straight lines only — adds perspective |

A homography is correct for a **planar scene viewed from different angles**. Two
ultrasound sweeps are *different cross-sections of a deforming 3D organ*, not two views
of a plane. The extra freedom lets it fold the image to fit outliers.

## The central finding

Measured on `phantom.pair()`, where the true transform is known:

| Model | Inliers | Reprojection RMSE | **True corner error** |
|---|---|---|---|
| rigid | 49/62 | 0.933 px | **0.421 px** |
| affine | 49/62 | 0.947 px | 0.695 px |
| homography | 48/62 | **0.839 px** | **2.136 px** |

**The homography has the best reprojection RMSE and the worst actual error, by a factor
of five.**

This is overfitting made visible. Eight degrees of freedom absorb noise that the correct
model cannot, so the residual the homography reports *on its own inliers* shrinks while
the transform drifts further from the truth.

Two consequences:

1. **RMSE can never be used to choose a model.** More parameters always lower it,
   including when they make the answer worse. Only a held-out ground truth separates
   them — and in the clinic there is not one.
2. Rigid scores best here because the phantom transform *is* a similarity (rotation +
   uniform scale + shift). The model matching the true generating process wins, which
   is the expected and reassuring ordering.

Affine remains the default: rigid is right for this phantom but too restrictive for
real probe motion, which includes out-of-plane shear.

### How the true error is measured

By how far the recovered transform moves the four image corners, in pixels — a
geometric error, interpretable directly — rather than a matrix norm, which is not.
The recovered matrix maps follow-up → baseline while the truth maps baseline →
follow-up, so it is compared against the truth *inverted*.

### Inlier ratio as the field-usable check

Without ground truth, `inlier_ratio` is the honest signal. Below ~0.3 the alignment
should not be trusted. The `/registration` page colours it on that threshold.

---

# CNN

## What is here

**YOLOv11n-seg**, in [`inference.py`](../backend/inference.py), trained on MMOTU via
[`data_prep/mmotu_to_yolo.py`](../data_prep/mmotu_to_yolo.py). Weights:
`ovascan_burnin1.pt`, ~6 MB.

A single-stage detector with a segmentation head. It predicts boxes, class scores and
mask coefficients in one pass, combining coefficients with learned prototype masks to
produce per-instance masks — as opposed to Mask R-CNN's two-stage propose-then-segment
design. The `n` (nano) scale is the smallest variant: the right choice for ~1.4k
training images, where a larger backbone would overfit.

## Training data preparation

MMOTU ships **semantic masks as PNG**, not polygon labels. `mmotu_to_yolo.py` traces
external contours, simplifies with `approxPolyDP` at 0.2% of perimeter, normalises to
[0,1] and writes Ultralytics segmentation labels.

Two decisions worth noting:

**All 8 tumour types collapse to one `lesion` class.** With ~1.4k images, eight-way
classification would have ~180 examples per class. One-class segmentation is the
defensible target; the fine-grained label is kept in the EHR record instead
([T1](T1-understanding-ehr.md)).

**The official `train.txt`/`val.txt` split is used, not a random one.** Those are split
by patient. A random split would put two images of the same ovary in train and val and
leak — inflating validation scores with no real generalisation.

## The interaction with restoration

The CNN is trained on **burn-in-removed** images, and inference applies the same
preprocessing. This matters more than it sounds: calipers sit inside the lesion, so a
model trained on marked images learns the calipers and collapses on unmarked ones. The
inpainting in [T4](T4-image-processing.md) is not cosmetic — it is what stops the model
learning a shortcut.

## Measured accuracy

Reproducible with [`backend/evaluate.py`](../backend/evaluate.py), which scores the
shipped checkpoint against MMOTU's own `_binary` masks, reading the archive directly:

```bash
python evaluate.py --zip "archive(1).zip" --n 120
```

120 validation images, patient-disjoint from training, `conf >= 0.25`:

| | |
|---|---|
| mean IoU | **0.704** (median 0.795) |
| mean Dice | **0.785** (median 0.886) |
| IoU ≥ 0.50 | 97/120 (81%) |
| IoU ≥ 0.75 | 76/120 (63%) |
| complete misses | 4 |
| sensitivity | 117/120 (98%) |
| latency, CPU | mean 320 ms, max 837 ms |

### The per-class result is the interesting one

| Class | n | mean IoU |
|---|---|---|
| serous cystadenoma | 19 | 0.876 |
| mucinous cystadenoma | 7 | 0.842 |
| high grade serous | 2 | 0.775 |
| teratoma | 36 | 0.769 |
| simple cyst | 2 | 0.751 |
| theca cell tumor | 9 | 0.748 |
| chocolate cyst | 25 | 0.731 |
| **normal ovary** | **20** | **0.310** |

Every lesion class lands between 0.73 and 0.88. **"Normal ovary" scores 0.310** — less
than half the next worst.

That is not a training failure, it is a **consequence of the one-class design**. Class 5's
ground truth is the *ovary*, not a lesion. Collapsing all eight MMOTU types into a single
`lesion` class therefore asks the model to segment two different targets under one label,
and it learns the lesion. The supervision is inconsistent.

Excluding class 5, mean IoU is **0.783** against 0.704 overall — so the 18% of the dataset
labelled "normal" costs ~0.08 on the headline number. Dropping those images, or giving
them their own class, is the obvious fix and is not currently done.

## What cannot be measured from this dataset

**Specificity and false-positive rate.** Every MMOTU image carries a non-empty mask,
class 5 included — sampled across 120 of the 267 normal-ovary images, not one was empty
(median mask area 13,533 px). There are no true negatives, so a detection on a "normal"
image is correct behaviour, not an error. Any false-positive rate computed here would be
counting correct detections as mistakes. Measuring one needs images with genuinely empty
ground truth, which MMOTU does not provide.

## Other limits

- One class, one dataset, one modality.
- No test-time augmentation, no ensembling, no calibration of the confidence scores —
  so the threshold slider is a raw score cut-off, not a probability.
- Two of the eight classes have only 2 validation images each, so their per-class means
  are indicative at best.

---

## How the three fit together

```
SIFT      finds correspondences between two frames   (geometry, unsupervised)
RANSAC    fits one transform, rejecting outliers     (robust estimation)
CNN       finds the lesion within one frame          (semantics, supervised)
```

A complete follow-up comparison uses all three: register with SIFT+RANSAC, segment each
frame with the CNN, then compare the measurements in the shared coordinate system. That
full three-stage pipeline **is** built, as `POST /image/register-and-segment`, and is
driven from the "Run end-to-end comparison" button on `/registration`.

Measured on the phantom pair: 34 of 49 ratio-tested matches accepted as inliers, both
frames segmented, and the registered follow-up compared against baseline at
**+1.59% area** and **+0.33% max diameter** — which is the number a clinician would
actually act on, and it is only meaningful because the frames share a coordinate system.

---

Previous: [T4](T4-image-processing.md) · Next: [T6](T6-visualization.md)
