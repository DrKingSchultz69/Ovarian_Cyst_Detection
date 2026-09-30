"""Image-level audit of MMOTU -- the T2/T3 checks that need the pixels.

WHY A SECOND SCRIPT
    mmotu_audit.py is a METADATA audit: it reads the zip index (names, sizes,
    CRC32) and the split files, and never decodes an image. That is what makes
    it cheap enough to prove the 1469 duplicate masks without extracting
    170 MB -- but it is also blind to everything inside the files.

    This pass opens the images. It answers the questions the metadata cannot:

      T2 standardization   are the images one size, one colour mode, one
                           bit depth? do masks agree with their image?
                           are mask values 0/1 or 0/255?
      T2 cleaning          are any files corrupt, truncated or undecodable?
      T3 redundancy        are any two SCANS near-identical? The masks are
                           proven byte-identical duplicates; whether the
                           images repeat is the more interesting question,
                           and needs perceptual hashing rather than CRC.
      T3 missing           are any masks empty -- a file that exists, has a
                           real byte size, and contains no annotation?

USAGE
    python mmotu_image_audit.py --zip "archive(1).zip"
    python mmotu_image_audit.py --zip "archive(1).zip" --n 400   # sample

    Decoding all 1469 images takes a few minutes; --n samples instead.
"""

from __future__ import annotations

import argparse
import collections
import io
import zipfile

import cv2
import numpy as np

ROOT = "MMOTU/OTU_2d/"

# Perceptual hash size. 8x8 DCT-based aHash over a 32x32 reduction: small
# enough that near-duplicates collide, large enough that different scans do
# not. Hamming distance <= 5 over 64 bits is the usual "near-duplicate" line.
PHASH_SIDE = 8
NEAR_DUPLICATE_BITS = 5


def _decode(data: bytes, flag: int) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), flag)


def phash(gray: np.ndarray) -> int:
    """Average-hash of a 8x8 DCT low-frequency block -> a 64-bit int.

    Robust to rescaling, mild compression and brightness shifts, which is
    what "the same scan saved twice" looks like. A byte-level hash would miss
    all of those, which is exactly why CRC found the duplicate masks but
    cannot answer this question.
    """
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:PHASH_SIDE, :PHASH_SIDE]
    flat = dct.flatten()[1:]  # drop DC: it is just overall brightness
    median = np.median(flat)
    bits = 0
    for i, v in enumerate(dct.flatten()[1:]):
        if v > median:
            bits |= 1 << i
    return bits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", dest="zip_path", required=True)
    ap.add_argument("--n", type=int, default=0, help="sample N images (0 = all)")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip_path)
    stems = sorted({
        n.split("/")[-1].rsplit(".", 1)[0]
        for n in z.namelist()
        if n.startswith(ROOT + "images/") and n.lower().endswith(".jpg")
    })
    if args.n and args.n < len(stems):
        rng = np.random.default_rng(0)
        stems = [stems[i] for i in sorted(rng.choice(len(stems), args.n, replace=False))]

    sizes: collections.Counter = collections.Counter()
    channels: collections.Counter = collections.Counter()
    mask_values: collections.Counter = collections.Counter()
    corrupt_images: list[str] = []
    corrupt_masks: list[str] = []
    empty_masks: list[str] = []
    dim_mismatch: list[tuple[str, tuple, tuple]] = []
    multi_component: list[tuple[str, int]] = []
    hashes: dict[str, int] = {}
    coverage: list[float] = []

    for stem in stems:
        try:
            img = _decode(z.read(f"{ROOT}images/{stem}.JPG"), cv2.IMREAD_UNCHANGED)
        except KeyError:
            continue
        if img is None:
            corrupt_images.append(stem)
            continue

        h, w = img.shape[:2]
        sizes[(w, h)] += 1
        channels[1 if img.ndim == 2 else img.shape[2]] += 1

        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hashes[stem] = phash(gray)

        try:
            mask = _decode(z.read(f"{ROOT}annotations/{stem}_binary.PNG"),
                           cv2.IMREAD_GRAYSCALE)
        except KeyError:
            mask = None
        if mask is None:
            corrupt_masks.append(stem)
            continue

        uniq = np.unique(mask)
        mask_values[tuple(int(v) for v in uniq[:4])] += 1

        if mask.shape[:2] != (h, w):
            dim_mismatch.append((stem, (w, h), (mask.shape[1], mask.shape[0])))
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        binary = (mask > 0).astype(np.uint8)
        area = int(binary.sum())
        if area == 0:
            empty_masks.append(stem)
            continue
        coverage.append(100.0 * area / (h * w))

        n_cc, _ = cv2.connectedComponents(binary)
        if n_cc - 1 > 1:
            multi_component.append((stem, n_cc - 1))

    # --- near-duplicate scans -------------------------------------------
    items = list(hashes.items())
    near: list[tuple[str, str, int]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = bin(items[i][1] ^ items[j][1]).count("1")
            if d <= NEAR_DUPLICATE_BITS:
                near.append((items[i][0], items[j][0], d))
    exact_hash = sum(1 for a, b, d in near if d == 0)

    # --- report -----------------------------------------------------------
    n = len(stems)
    print(f"\nimages decoded                {len(hashes)}/{n}")

    print("\n--- T2  standardization: is the data uniform? ---")
    print(f"  distinct image sizes        {len(sizes)}")
    for (w, h), c in sizes.most_common(6):
        print(f"      {w:>5} x {h:<5}  {c:>5} images  ({100*c/max(n,1):.1f}%)")
    if len(sizes) > 6:
        print(f"      ... and {len(sizes)-6} more")
    print(f"  channel counts              {dict(channels)}   (3 = BGR, 1 = greyscale)")
    print(f"  distinct mask value sets    {len(mask_values)}")
    for vals, c in mask_values.most_common(4):
        print(f"      {list(vals)}  ->  {c} masks")

    print("\n--- T2  cleaning: is anything broken? ---")
    print(f"  undecodable images          {len(corrupt_images)}")
    print(f"  undecodable / absent masks  {len(corrupt_masks)}")
    print(f"  mask != image dimensions    {len(dim_mismatch)}")
    for stem, i_wh, m_wh in dim_mismatch[:5]:
        print(f"      {stem}: image {i_wh}  mask {m_wh}")

    print("\n--- T3  redundancy: do any SCANS repeat? ---")
    print(f"  pairs compared              {len(items)*(len(items)-1)//2:,}")
    print(f"  identical perceptual hash   {exact_hash}")
    print(f"  near-duplicate (<= {NEAR_DUPLICATE_BITS} bits)  {len(near)}")

    # THE QUESTION THAT MATTERS. mmotu_audit.py proves train and val share no
    # ID, and that is what "patient-disjoint" is usually taken to mean. But two
    # DIFFERENT ids can still be the same scan -- consecutive frames of one
    # sweep, or the same study saved twice. If such a pair straddles the split,
    # the model sees a val image in training and every val metric is inflated.
    # ID-disjointness cannot detect that; only pixels can.
    train = {l.split()[0] for l in z.read(ROOT + "train.txt").decode().splitlines() if l.strip()}
    val = {l.split()[0] for l in z.read(ROOT + "val.txt").decode().splitlines() if l.strip()}

    def side(stem: str) -> str:
        return "train" if stem in train else "val" if stem in val else "?"

    crossing = [(a, b, d) for a, b, d in near if side(a) != side(b) and "?" not in (side(a), side(b))]

    for a, b, d in sorted(near, key=lambda t: t[2])[:12]:
        flag = "  <-- CROSSES SPLIT" if (a, b, d) in crossing else ""
        print(f"      {a} ({side(a)}) ~ {b} ({side(b)})   hamming {d}{flag}")
    if not near:
        print("      none -- every scan is perceptually distinct.")

    print(f"\n  near-duplicate pairs crossing train/val   {len(crossing)}")
    if crossing:
        print("      ^ LEAKAGE: these are effectively the same scan on both")
        print("        sides of the split. ID-level disjointness did not catch it.")
    else:
        print("      none -- no near-duplicate pair straddles the split, so the")
        print("      official split survives a pixel-level check, not just an ID one.")

    print("\n--- T3  missing / degenerate annotation ---")
    print(f"  empty masks (0 pixels set)  {len(empty_masks)}")
    if empty_masks[:8]:
        print(f"      {empty_masks[:8]}")
    print(f"  masks with >1 component     {len(multi_component)}")
    for stem, k in multi_component[:5]:
        print(f"      {stem}: {k} components")
    if coverage:
        cov = np.array(coverage)
        print(f"  lesion coverage of frame    min {cov.min():.2f}%  "
              f"median {np.median(cov):.2f}%  max {cov.max():.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
