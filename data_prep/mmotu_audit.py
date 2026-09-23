"""Audit the MMOTU OTU_2d release as a DATA QUALITY exercise -- T1, T2, T3 on
the real dataset, alongside the synthetic cohort in backend/ehr/.

WHY THIS EXISTS
    backend/ehr/ covers T1-T3 against a synthetic cohort, because judging a
    repair needs the uncorrupted original and no real extract ships one. The
    fair criticism of that is that it is fabricated. This module answers it:
    MMOTU carries its own small structured record -- image, mask, class label,
    split assignment -- and that record has real defects, which are measured
    here rather than asserted.

    The two are complementary, and the split is deliberate:

      real (here)        finds the defects that actually exist, and proves
                         the checks run against something nobody planted
      synthetic (ehr/)   supplies the defects MMOTU does NOT have -- unit
                         mismatches, date drift, impossible values, three
                         missingness mechanisms -- so cleaning and imputation
                         can be demonstrated and scored at all

    Run this first. It reads the zip index only, so it needs no extraction and
    no 170 MB on disk.

        python mmotu_audit.py --zip "archive(1).zip"
        python mmotu_audit.py --root path/to/MMOTU/OTU_2d

HEADLINE FINDING
    Every `_binary_binary.PNG` is a byte-identical copy of the corresponding
    `_binary.PNG`. All 1469 of them, zero exceptions, checked by CRC32 and
    size. That is 33.3% of the annotation files in the release -- 3.6 MB of
    pure redundancy. Real T3 material, found in a published dataset.

    The byte count is modest because these are small binary masks; the point
    is the proportion, not the megabytes. A third of the annotation directory
    can be deleted with no information loss whatsoever.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import os
import zipfile
from typing import Iterable

# MMOTU class id -> the dataset's own tumour-type label. The mapping onto
# ICD-10 lives in backend/ehr/vocab.py; keeping the local labels here and the
# standard codes there is the T2 separation, not duplication.
MMOTU_CLASSES = {
    0: "chocolate cyst",
    1: "serous cystadenoma",
    2: "teratoma",
    3: "theca cell tumor",
    4: "simple cyst",
    5: "normal ovary",
    6: "mucinous cystadenoma",
    7: "high grade serous",
}


@dataclasses.dataclass
class Entry:
    """One annotation file, as the index reports it. `crc` is None for a
    filesystem source, where the hash is computed instead."""

    name: str
    size: int
    crc: int | None


@dataclasses.dataclass
class Audit:
    images: set[str]
    plain: set[str]
    binary: set[str]
    binary_binary: set[str]
    duplicate_pairs: int
    differing_pairs: int
    redundant_bytes: int
    train_ids: set[str]
    val_ids: set[str]
    overlap: set[str]
    class_counts: dict[int, int]
    labelled: int
    missing_images: set[str]
    missing_masks: set[str]
    unsplit: set[str]


# ---------------------------------------------------------------------------
# Sources: a zip index, or an extracted folder
# ---------------------------------------------------------------------------


def _from_zip(path: str, root: str) -> tuple[set[str], list[Entry], dict[str, str]]:
    archive = zipfile.ZipFile(path)
    images: set[str] = set()
    anns: list[Entry] = []
    texts: dict[str, str] = {}

    for info in archive.infolist():
        name = info.filename
        if not name.startswith(root):
            continue
        base = os.path.basename(name)
        lower = base.lower()

        if name.startswith(root + "images/") and lower.endswith(".jpg"):
            images.add(os.path.splitext(base)[0])
        elif name.startswith(root + "annotations/") and lower.endswith(".png"):
            anns.append(Entry(base, info.file_size, info.CRC))
        elif lower.endswith(".txt") and "/" not in name[len(root):]:
            texts[base] = archive.read(name).decode("utf-8", "replace")

    return images, anns, texts


def _from_dir(root: str) -> tuple[set[str], list[Entry], dict[str, str]]:
    images = {
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(root, "images"))
        if f.lower().endswith(".jpg")
    }

    ann_dir = os.path.join(root, "annotations")
    anns = [
        Entry(f, os.path.getsize(os.path.join(ann_dir, f)), None)
        for f in os.listdir(ann_dir)
        if f.lower().endswith(".png")
    ]

    texts = {
        f: open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        for f in os.listdir(root)
        if f.lower().endswith(".txt")
    }
    return images, anns, texts


def _digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------


def _split_ids(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if line.strip()]


def audit(zip_path: str | None = None, root_dir: str | None = None,
          zip_root: str = "MMOTU/OTU_2d/") -> Audit:
    if zip_path:
        images, anns, texts = _from_zip(zip_path, zip_root)
        ann_dir = None
    elif root_dir:
        images, anns, texts = _from_dir(root_dir)
        ann_dir = os.path.join(root_dir, "annotations")
    else:
        raise ValueError("pass either zip_path or root_dir")

    # --- T3: redundant data --------------------------------------------
    plain: set[str] = set()
    binary: set[str] = set()
    binbin: set[str] = set()
    by_name: dict[str, Entry] = {}

    for entry in anns:
        stem = os.path.splitext(entry.name)[0]
        by_name[entry.name] = entry
        if stem.endswith("_binary_binary"):
            binbin.add(stem[: -len("_binary_binary")])
        elif stem.endswith("_binary"):
            binary.add(stem[: -len("_binary")])
        else:
            plain.add(stem)

    same = differ = 0
    redundant_bytes = 0
    for stem in sorted(binary & binbin):
        a = by_name.get(f"{stem}_binary.PNG")
        b = by_name.get(f"{stem}_binary_binary.PNG")
        if not (a and b):
            continue

        if a.crc is not None and b.crc is not None:
            identical = a.crc == b.crc and a.size == b.size
        else:
            # Filesystem source: size first, hash only when it could matter.
            identical = a.size == b.size and (
                ann_dir is None
                or _digest(os.path.join(ann_dir, a.name))
                == _digest(os.path.join(ann_dir, b.name))
            )

        if identical:
            same += 1
            redundant_bytes += b.size
        else:
            differ += 1

    # --- T1/T2: the record and its coding -------------------------------
    train = {r[0] for r in _split_ids(texts.get("train.txt", ""))}
    val = {r[0] for r in _split_ids(texts.get("val.txt", ""))}

    class_counts: collections.Counter[int] = collections.Counter()
    for key in ("train_cls.txt", "val_cls.txt"):
        for row in _split_ids(texts.get(key, "")):
            if len(row) >= 2 and row[1].lstrip("-").isdigit():
                class_counts[int(row[1])] += 1

    split_ids = train | val
    return Audit(
        images=images,
        plain=plain,
        binary=binary,
        binary_binary=binbin,
        duplicate_pairs=same,
        differing_pairs=differ,
        redundant_bytes=redundant_bytes,
        train_ids=train,
        val_ids=val,
        overlap=train & val,
        class_counts=dict(class_counts),
        labelled=sum(class_counts.values()),
        missing_images=split_ids - images,
        missing_masks=images - plain,
        unsplit=images - split_ids,
    )


def report(a: Audit) -> str:
    lines: list[str] = []

    def head(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    head("T1  what the release actually contains")
    lines.append(f"  images (.JPG)                 {len(a.images)}")
    lines.append(f"  plain masks                   {len(a.plain)}")
    lines.append(f"  _binary masks                 {len(a.binary)}")
    lines.append(f"  _binary_binary masks          {len(a.binary_binary)}")
    lines.append(f"  labelled rows (cls files)     {a.labelled}")
    lines.append("  record shape: image <-> mask <-> class label <-> split assignment")

    head("T2  coding -- the dataset's local vocabulary")
    total = max(a.labelled, 1)
    for cid in sorted(a.class_counts):
        n = a.class_counts[cid]
        lines.append(
            f"  {cid}  {MMOTU_CLASSES.get(cid, '?'):22s} {n:5d}  {100 * n / total:5.1f}%"
        )
    if a.class_counts:
        worst = min(a.class_counts.values())
        best = max(a.class_counts.values())
        lines.append(f"  imbalance ratio (max/min)     {best / max(worst, 1):.1f}x")
        lines.append("  these 8 local labels map onto 5 ICD-10 codes in backend/ehr/vocab.py")

    head("T3  redundant data")
    lines.append(f"  _binary / _binary_binary pairs        {len(a.binary & a.binary_binary)}")
    lines.append(f"  byte-identical (CRC + size)           {a.duplicate_pairs}")
    lines.append(f"  differing                             {a.differing_pairs}")
    lines.append(f"  redundant payload                     {a.redundant_bytes / 1e6:.1f} MB")
    if a.binary and a.duplicate_pairs == len(a.binary & a.binary_binary):
        share = 100 * len(a.binary_binary) / max(
            len(a.plain) + len(a.binary) + len(a.binary_binary), 1
        )
        lines.append(
            f"  EVERY _binary_binary duplicates its _binary: {share:.1f}% of the "
            f"annotation files are removable with no information loss"
        )

    head("T3  missing data and split integrity")
    lines.append(f"  split ids with no image file          {len(a.missing_images)}")
    lines.append(f"  images with no mask                   {len(a.missing_masks)}")
    lines.append(f"  images in no split                    {len(a.unsplit)}")
    lines.append(f"  train n val overlap (leakage)         {len(a.overlap)}")
    if not (a.missing_images or a.missing_masks or a.unsplit or a.overlap):
        lines.append("  -> complete and non-leaking. Nothing to repair, which is")
        lines.append("     itself the finding: the CHECKS are the deliverable, and")
        lines.append("     a clean result is only meaningful because they ran.")

    head("why the synthetic cohort still exists")
    lines.append("  MMOTU has no units, no dates, no free-text diagnoses, no patient")
    lines.append("  identifiers and no missing values -- so it cannot demonstrate unit")
    lines.append("  conversion, date parsing, record linkage or imputation at all.")
    lines.append("  backend/ehr/ supplies those against a cohort whose defects are")
    lines.append("  known, and joins to these images on imaging_studies.scan_stem.")

    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", dest="zip_path", help="path to the MMOTU archive")
    ap.add_argument("--root", dest="root_dir", help="path to an extracted MMOTU/OTU_2d")
    ap.add_argument("--zip-root", default="MMOTU/OTU_2d/",
                    help="prefix inside the archive (default: MMOTU/OTU_2d/)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.zip_path and not args.root_dir:
        ap.error("pass --zip or --root")

    print(report(audit(args.zip_path, args.root_dir, args.zip_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
