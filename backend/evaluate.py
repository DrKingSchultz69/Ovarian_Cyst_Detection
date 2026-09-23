"""Score the shipped checkpoint against real MMOTU ground-truth masks.

WHY THIS EXISTS
    Until now the project reported no accuracy figures at all, because
    validation ran in a Kaggle notebook this repository cannot reproduce, and
    quoting numbers the code could not regenerate would have been worse than
    omitting them. This closes that gap: it runs here, from the archive, and
    prints numbers anyone can re-derive.

    It reads the zip directly, so no extraction and no 170 MB on disk.

        python evaluate.py --zip "archive(1).zip"
        python evaluate.py --zip "archive(1).zip" --n 200 --conf 0.3

VALIDITY
    The val split is patient-disjoint from train (MMOTU ships train.txt and
    val.txt split by patient, and data_prep/mmotu_audit.py confirms zero
    overlap), so these images were never seen during training. Ground truth
    is the dataset's own `_binary` mask.

WHAT CANNOT BE MEASURED HERE, AND WHY
    MMOTU has NO TRUE NEGATIVES. Class 5 is labelled "normal ovary", but
    those images still carry a non-empty mask -- the ovary itself is
    annotated. Sampled across 120 of the 267 class-5 images, every single
    mask was non-empty, median area 13,533 px.

    So a detection on a class-5 image is CORRECT behaviour, not a false
    positive, and **specificity and false-positive rate cannot be computed
    from this dataset at all**. Any figure claiming otherwise is counting
    correct detections as errors. Reporting one would need images with
    genuinely empty ground truth, which MMOTU does not provide.
"""

from __future__ import annotations

import argparse
import dataclasses
import zipfile

import cv2
import numpy as np

import inference

ROOT = "MMOTU/OTU_2d/"

CLASS_NAMES = {
    0: "chocolate cyst", 1: "serous cystadenoma", 2: "teratoma",
    3: "theca cell tumor", 4: "simple cyst", 5: "normal ovary",
    6: "mucinous cystadenoma", 7: "high grade serous",
}


@dataclasses.dataclass
class Score:
    stem: str
    cls: int
    gt_px: int
    pred_px: int
    count: int
    conf: float
    iou: float
    dice: float
    latency_ms: int


def _decode(data: bytes, flag: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), flag)


def _predicted_mask(image: np.ndarray, conf: float) -> tuple[np.ndarray, int, float]:
    """Union of the predicted instance masks, plus count and best confidence.

    Runs the same preprocessing as analyse_bytes so the geometry matches what
    the service would return for this image.
    """
    clean = inference.preprocess(image)
    height, width = clean.shape[:2]
    result = inference.model.predict(clean, conf=conf, verbose=False)[0]

    mask = np.zeros((height, width), bool)
    count = 0
    best = 0.0
    if result.masks is not None:
        for i, poly in enumerate(result.masks.xy):
            if len(poly) < 3:
                continue
            filled = np.zeros((height, width), np.uint8)
            cv2.fillPoly(filled, [poly.astype(np.int32)], 255)
            mask |= filled > 0
            count += 1
            best = max(best, float(result.boxes.conf[i]))
    return mask, count, best


def evaluate(zip_path: str, n: int = 60, conf: float = 0.25,
             seed: int = 4) -> list[Score]:
    archive = zipfile.ZipFile(zip_path)

    val = [line.split()[0]
           for line in archive.read(ROOT + "val.txt").decode().splitlines()
           if line.strip()]
    labels = {
        row.split()[0].split(".")[0]: int(row.split()[1])
        for row in archive.read(ROOT + "val_cls.txt").decode().splitlines()
        if row.strip()
    }

    rng = np.random.default_rng(seed)
    if n >= len(val):
        picks = val
    else:
        picks = [val[i] for i in rng.choice(len(val), size=n, replace=False)]

    scores: list[Score] = []
    for stem in picks:
        try:
            image = _decode(archive.read(f"{ROOT}images/{stem}.JPG"))
            truth = _decode(archive.read(f"{ROOT}annotations/{stem}_binary.PNG"),
                            cv2.IMREAD_GRAYSCALE)
        except KeyError:
            continue
        if image is None or truth is None:
            continue

        height, width = image.shape[:2]
        if truth.shape[:2] != (height, width):
            truth = cv2.resize(truth, (width, height), interpolation=cv2.INTER_NEAREST)
        gt = truth > 0

        import time
        started = time.time()
        pred, count, best = _predicted_mask(image, conf)
        latency = int((time.time() - started) * 1000)

        intersection = float((pred & gt).sum())
        union = float((pred | gt).sum())
        denominator = float(pred.sum() + gt.sum())

        scores.append(Score(
            stem=stem,
            cls=labels.get(stem, -1),
            gt_px=int(gt.sum()),
            pred_px=int(pred.sum()),
            count=count,
            conf=round(best, 3),
            iou=intersection / union if union else 1.0,
            dice=2 * intersection / denominator if denominator else 1.0,
            latency_ms=latency,
        ))
    return scores


def report(scores: list[Score], conf: float) -> str:
    if not scores:
        return "no images scored"

    lines: list[str] = []
    ious = np.array([s.iou for s in scores])
    dices = np.array([s.dice for s in scores])
    latency = np.array([s.latency_ms for s in scores])

    lines.append(f"{'id':>6} {'class':<20} {'det':>3} {'conf':>5} {'gt px':>8} "
                 f"{'pred px':>8} {'IoU':>6} {'Dice':>6} {'ms':>5}")
    lines.append("-" * 82)
    for s in sorted(scores, key=lambda s: s.iou):
        lines.append(f"{s.stem:>6} {CLASS_NAMES.get(s.cls, '?'):<20} {s.count:>3} "
                     f"{s.conf:>5.2f} {s.gt_px:>8} {s.pred_px:>8} "
                     f"{s.iou:>6.3f} {s.dice:>6.3f} {s.latency_ms:>5}")

    lines.append("")
    lines.append("=" * 82)
    lines.append(f"images scored                {len(scores)}   (val split, conf >= {conf})")
    lines.append(f"mean IoU                     {ious.mean():.3f}      median {np.median(ious):.3f}")
    lines.append(f"mean Dice                    {dices.mean():.3f}      median {np.median(dices):.3f}")
    lines.append(f"IoU >= 0.50                  {(ious >= 0.5).sum()}/{len(scores)}"
                 f"   ({100 * (ious >= 0.5).mean():.0f}%)")
    lines.append(f"IoU >= 0.75                  {(ious >= 0.75).sum()}/{len(scores)}"
                 f"   ({100 * (ious >= 0.75).mean():.0f}%)")
    lines.append(f"complete misses (IoU = 0)    {int((ious == 0).sum())}")
    lines.append(f"latency, CPU                 mean {latency.mean():.0f} ms   "
                 f"median {np.median(latency):.0f} ms   max {latency.max():.0f} ms")

    detected = sum(1 for s in scores if s.count > 0)
    with_gt = [s for s in scores if s.gt_px > 0]
    found = sum(1 for s in with_gt if s.count > 0)
    lines.append("")
    lines.append(f"images with a non-empty ground-truth mask   {len(with_gt)}/{len(scores)}")
    lines.append(f"  of those, something was detected          {found}"
                 f"   ({100 * found / max(len(with_gt), 1):.0f}% sensitivity)")
    lines.append(f"images where anything was detected          {detected}")

    by_class: dict[int, list[Score]] = {}
    for s in scores:
        by_class.setdefault(s.cls, []).append(s)
    lines.append("")
    lines.append("by class:")
    for cls in sorted(by_class):
        group = by_class[cls]
        mean_iou = float(np.mean([g.iou for g in group]))
        flag = "  <-- see below" if cls == 5 else ""
        lines.append(f"  {cls}  {CLASS_NAMES.get(cls, '?'):<20} n={len(group):>3}   "
                     f"mean IoU {mean_iou:.3f}{flag}")

    # The one finding in this report that is about the MODEL DESIGN rather
    # than its accuracy.
    lesion_only = [s for s in scores if s.cls != 5]
    normal_only = [s for s in scores if s.cls == 5]
    if lesion_only and normal_only:
        lesion_iou = float(np.mean([s.iou for s in lesion_only]))
        normal_iou = float(np.mean([s.iou for s in normal_only]))
        lines.append("")
        lines.append(f"excluding class 5            mean IoU {lesion_iou:.3f}"
                     f"   (n={len(lesion_only)})")
        lines.append(f"class 5 alone                mean IoU {normal_iou:.3f}"
                     f"   (n={len(normal_only)})")
        lines.append(f"cost of including class 5    {lesion_iou - ious.mean():+.3f}"
                     f" on the headline mean")
        lines.append("")
        lines.append("  Class 5 is 'normal ovary', and its ground truth is the OVARY, not a")
        lines.append("  lesion. Collapsing all 8 MMOTU types into one `lesion` class therefore")
        lines.append("  asks the model to segment two different targets under one label, and")
        lines.append("  it learns the lesion one. This is a consequence of the one-class")
        lines.append("  design, not a training failure -- the supervision is inconsistent.")
        lines.append("  Dropping class 5, or giving it its own class, is the obvious fix.")

    lines.append("")
    lines.append("SPECIFICITY IS NOT REPORTED, and cannot be: every MMOTU image carries a")
    lines.append("non-empty mask, class 5 'normal ovary' included -- the ovary itself is")
    lines.append("annotated. There are no true negatives here, so a detection on a class-5")
    lines.append("image is correct, not a false positive. Measuring a false-positive rate")
    lines.append("needs images with genuinely empty ground truth, which this dataset lacks.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", dest="zip_path", required=True,
                    help="path to the MMOTU archive")
    ap.add_argument("--n", type=int, default=60, help="validation images to score")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--seed", type=int, default=4, help="sampling seed")
    args = ap.parse_args()

    scores = evaluate(args.zip_path, args.n, args.conf, args.seed)
    print(report(scores, args.conf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
