"""MMOTU OTU_2d -> YOLOv11-seg dataset.

MMOTU ships semantic masks as PNG, not polygon labels. This traces each mask's
external contours, normalises them to [0,1] and writes Ultralytics segmentation
label files. All 8 tumour types collapse to a single `lesion` class, matching
the one-class model.

The official train.txt / val.txt are used for the split. A random split would
certainly leak, so the official one is the better choice -- but it is NOT
leak-free, and an earlier version of this note claimed it was.

MEASURED (data_prep/mmotu_image_audit.py, all 1469 images):
    31 image pairs share an identical perceptual hash
    106 pairs are near-duplicates (Hamming <= 5 of a 64-bit pHash)
    48 of those pairs straddle the train/val boundary
    44 of the 469 validation images -- 9.4% -- have a near-duplicate in train

    Verified against pixels, not just the hash: those pairs run MAE 5-13 on a
    0-255 scale with correlation 0.88-0.98, at slightly different crops. They
    are the same scan stored twice under consecutive ids (1268/1269,
    1430/1431, 144/146 ...), not hash collisions.

The ids are disjoint, which is all an id-level check can confirm; the IMAGES
are not. Any metric measured on the official val split is therefore optimistic
by an unknown amount, and the honest fix is to drop the contaminated val ids
before evaluating rather than to re-split at random.

Kaggle:  exec(open('mmotu_to_yolo.py').read())   or paste as a cell.
Local:   python mmotu_to_yolo.py --root <MMOTU/OTU_2d> --out <ova_yolo>
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np

DEFAULT_ROOT = ("/kaggle/input/datasets/orvile/mmotu-ovarian-ultrasound-images-dataset"
                "/MMOTU/OTU_2d")
DEFAULT_OUT = "/kaggle/working/ova_yolo"

MIN_CONTOUR_AREA = 60      # px; drops speckle that would become 3-point labels
EPS_FRACTION = 0.002       # approxPolyDP tolerance as a fraction of perimeter


def read_split(path: str) -> list[str]:
    """Each line starts with a filename or id; the rest (class label) is ignored
    because every tumour type maps to the same `lesion` class."""
    ids = []
    with open(path) as fh:
        for line in fh:
            token = line.strip().split()
            if token:
                ids.append(os.path.splitext(token[0])[0])
    return ids


def mask_to_polygons(mask: np.ndarray, w: int, h: int) -> list[np.ndarray]:
    """Binary mask -> list of flat, normalised [x1,y1,x2,y2,...] polygons."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        eps = EPS_FRACTION * cv2.arcLength(c, True)
        pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue
        pts[:, 0] /= w
        pts[:, 1] /= h
        polys.append(np.clip(pts, 0.0, 1.0).reshape(-1))
    return polys


def convert(root: str = DEFAULT_ROOT, out: str = DEFAULT_OUT) -> str:
    img_dir = os.path.join(root, "images")
    ann_dir = os.path.join(root, "annotations")

    for split in ("train", "val"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    totals = {}
    for split in ("train", "val"):
        ids = read_split(os.path.join(root, f"{split}.txt"))
        written = empty = missing = 0

        for stem in ids:
            src_img = os.path.join(img_dir, f"{stem}.JPG")
            src_msk = os.path.join(ann_dir, f"{stem}_binary.PNG")
            if not (os.path.exists(src_img) and os.path.exists(src_msk)):
                missing += 1
                continue

            image = cv2.imread(src_img, cv2.IMREAD_COLOR)
            mask = cv2.imread(src_msk, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                missing += 1
                continue

            h, w = image.shape[:2]
            # Masks are usually the same size as the image, but not always.
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Stored as 0/1; threshold rather than assuming the scale.
            binary = (mask > 0).astype(np.uint8)
            polys = mask_to_polygons(binary, w, h)

            shutil.copy(src_img, os.path.join(out, "images", split, f"{stem}.jpg"))
            label_path = os.path.join(out, "labels", split, f"{stem}.txt")
            with open(label_path, "w") as fh:
                for poly in polys:
                    fh.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")

            # An empty label file is legal in YOLO -- it means "background only"
            # and still contributes negative supervision. Counted, not dropped.
            written += 1
            empty += int(not polys)

        totals[split] = (written, empty, missing)
        print(f"{split}: {written} images written, {empty} with no polygon, "
              f"{missing} missing/unreadable")

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(
            f"path: {out}\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names:\n"
            "  0: lesion\n"
        )
    print(f"\ndata.yaml -> {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    convert(args.root, args.out)
