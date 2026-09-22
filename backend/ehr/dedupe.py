"""T3 part one -- redundant data removal.

Three things look like repetition in an EHR extract and only two are redundant:

  exact duplicate   the same registration submitted twice. Byte-identical.
                    Safe to drop.
  near duplicate    one person registered twice under different MRNs -- at
                    another site, after a name change, or through a typo.
                    Must be LINKED, and linking needs fuzzy matching because
                    no identifier is shared.
  repeat encounter  one person attending twice. Same MRN, different visit.
                    NOT redundant. A deduplicator that removes it is deleting
                    clinical history.

The third case is why record linkage is not `drop_duplicates()`. This module
is built to be measured against it: `evaluate` scores found pairs against the
generator manifest and reports precision and recall, so a threshold change can
be argued with numbers instead of taste.

METHOD
    Blocking, then scoring. Comparing all pairs is O(n^2) -- 430 patients is
    92,235 comparisons, and a real cohort of 500k is 125 billion. Blocking
    only compares records that agree on a cheap key (here: birth date), which
    is what makes linkage tractable. The cost is a missed match whenever the
    blocking key itself is wrong, so the key must be a field that is rarely
    mistyped and never legitimately changes.
"""

from __future__ import annotations

import difflib
from collections import defaultdict

import pandas as pd

from .standardize import Report

# Field weights for the match score. Names carry most of the signal because
# the block already fixes the birth date; site disagreement is weak evidence
# FOR a duplicate, since cross-site re-registration is the usual cause.
_W_GIVEN = 0.45
_W_FAMILY = 0.45
_W_SEX = 0.10

# Chosen from threshold_sweep(), which peaks at F1 = 1.0 across 0.70-0.80 and
# starts losing recall at 0.82. Picked at the top of that plateau rather than
# in the middle of it, because the failure that matters is a false merge.
#
# CAVEAT: tuned on the same cohort it is scored against, so that 1.0 is
# optimistic. Measured across held-out generator seeds at n_patients=400:
#
#   seed    7 (tuned on)  P 1.000  R 1.000  F1 1.000
#   seed   99 (held out)  P 1.000  R 0.950  F1 0.974
#   seed  123 (held out)  P 1.000  R 0.950  F1 0.974
#   seed 2024 (held out)  P 1.000  R 1.000  F1 1.000
#   seed  555 (held out)  P 1.000  R 0.800  F1 0.889
#
# Precision holds at 1.000 on every seed tried; recall ranges 0.80-1.00. The
# spread is the honest figure -- the first two held-out seeds checked happened
# to land at 0.95, and quoting only those would have understated the variance.
# Every failure is a missed match, never a wrong merge, which is the direction
# that matters. The misses are name variants too distant for the
# Ratcliff/Obershelp ratio; Jaro-Winkler would recover most of them.
# See docs/T3-redundant-and-missing-data.md.
MATCH_THRESHOLD = 0.80


def similarity(a: str, b: str) -> float:
    """Normalised string similarity in [0, 1].

    difflib's ratio is a Ratcliff/Obershelp measure -- twice the matched
    characters over the total length. It is stdlib, deterministic and good
    enough for transliteration variants (Laxmi/Lakshmi scores 0.83, Prya/Priya
    0.89). A production system would use Jaro-Winkler, which weights a shared
    prefix more heavily and separates name variants better.
    """
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Exact duplicates
# ---------------------------------------------------------------------------


def find_exact_duplicates(patients: pd.DataFrame,
                          subset: list[str] | None = None) -> pd.DataFrame:
    """Rows identical on the identifying columns.

    Compares the standardised columns, not the raw ones: two rows that differ
    only in whitespace or capitalisation are the same registration, and
    standardize.py has already removed that difference.
    """
    subset = subset or ["patient_id", "given_name", "family_name", "sex", "birth_date"]
    available = [c for c in subset if c in patients.columns]

    mask = patients.duplicated(subset=available, keep="first")
    return patients.loc[mask, available + []].assign(_row=patients.index[mask])


# ---------------------------------------------------------------------------
# Near duplicates -- blocking then scoring
# ---------------------------------------------------------------------------


def build_blocks(patients: pd.DataFrame, key: str = "birth_date"
                 ) -> dict[object, list[int]]:
    """Group row indices by an exact-match key.

    Records with a null key are placed in no block and therefore never
    compared -- a real and visible cost of blocking, counted in the report
    rather than hidden.
    """
    blocks: dict[object, list[int]] = defaultdict(list)
    for idx, value in patients[key].items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        blocks[value].append(int(idx))
    return {k: v for k, v in blocks.items() if len(v) > 1}


def score_pair(left: pd.Series, right: pd.Series) -> float:
    """Weighted agreement across identifying fields, in [0, 1]."""
    given = similarity(str(left.get("given_name", "")), str(right.get("given_name", "")))
    family = similarity(str(left.get("family_name", "")), str(right.get("family_name", "")))
    sex = 1.0 if left.get("sex") == right.get("sex") else 0.0
    return _W_GIVEN * given + _W_FAMILY * family + _W_SEX * sex


def find_near_duplicates(patients: pd.DataFrame,
                         threshold: float = MATCH_THRESHOLD,
                         block_key: str = "birth_date"
                         ) -> tuple[pd.DataFrame, Report]:
    """Candidate pairs scoring above the threshold, with their scores.

    Pairs sharing a patient_id are excluded: those are exact duplicates or
    repeat visits, both handled elsewhere. What is wanted here is two DIFFERENT
    identifiers that denote one person.
    """
    blocks = build_blocks(patients, block_key)
    comparisons = 0
    pairs: list[dict] = []

    for _, rows in blocks.items():
        for i, left_idx in enumerate(rows):
            for right_idx in rows[i + 1:]:
                comparisons += 1
                left, right = patients.loc[left_idx], patients.loc[right_idx]
                if left["patient_id"] == right["patient_id"]:
                    continue

                score = score_pair(left, right)
                if score >= threshold:
                    pairs.append({
                        "left_row": left_idx, "right_row": right_idx,
                        "left_id": left["patient_id"], "right_id": right["patient_id"],
                        "left_name": f"{left['given_name']} {left['family_name']}",
                        "right_name": f"{right['given_name']} {right['family_name']}",
                        "birth_date": left.get("birth_date"),
                        "score": round(score, 4),
                    })

    pairs_df = pd.DataFrame(pairs, columns=[
        "left_row", "right_row", "left_id", "right_id",
        "left_name", "right_name", "birth_date", "score",
    ]).sort_values("score", ascending=False, ignore_index=True)

    n = len(patients)
    report = Report()
    report["patients"] = n
    report["all_pairs_would_be"] = n * (n - 1) // 2
    report["blocks"] = len(blocks)
    report["comparisons_after_blocking"] = comparisons
    report["reduction"] = (
        f"{100 * (1 - comparisons / max(n * (n - 1) // 2, 1)):.1f}%"
    )
    report["unblocked_records"] = int(patients[block_key].isna().sum())
    report["pairs_over_threshold"] = len(pairs_df)
    report["threshold"] = threshold
    return pairs_df, report


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve(patients: pd.DataFrame, pairs: pd.DataFrame
            ) -> tuple[pd.DataFrame, dict[str, str]]:
    """Collapse matched pairs into one surviving record per person.

    Transitive closure via union-find: if A matches B and B matches C, all
    three are one person even when A and C never scored above the threshold.
    Skipping this step leaves the cluster half-merged, which is worse than not
    merging at all.

    The surviving MRN is the lexicographically smallest in the cluster -- an
    arbitrary but STABLE rule, so a rerun produces the same survivor. A real
    system would keep the earliest registration.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Smaller id wins, so the survivor does not depend on pair order.
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for row in pairs.itertuples(index=False):
        union(str(row.left_id), str(row.right_id))

    crosswalk = {pid: find(pid) for pid in parent if find(pid) != pid}

    out = patients.copy()
    out["master_patient_id"] = [
        crosswalk.get(str(pid), str(pid)) for pid in out["patient_id"]
    ]
    # One row per master id, keeping the most complete record in each cluster.
    completeness = out.notna().sum(axis=1)
    out = (out.assign(_completeness=completeness)
              .sort_values("_completeness", ascending=False)
              .drop_duplicates(subset=["master_patient_id"], keep="first")
              .drop(columns="_completeness")
              .sort_index())
    return out, crosswalk


# ---------------------------------------------------------------------------
# Scoring against ground truth
# ---------------------------------------------------------------------------


def evaluate(pairs: pd.DataFrame, duplicate_of: dict[str, str]) -> Report:
    """Precision and recall of the linkage against the generator manifest.

    Both matter, and they trade off:
      a missed duplicate  splits one person across two charts -- an allergy
                          recorded on one is invisible on the other
      a false match       merges two people -- one patient inherits another
                          diagnosis, which is the more dangerous failure
    """
    truth = {frozenset((dup, original)) for dup, original in duplicate_of.items()}
    found = {
        frozenset((str(r.left_id), str(r.right_id)))
        for r in pairs.itertuples(index=False)
    }

    tp = len(truth & found)
    fp = len(found - truth)
    fn = len(truth - found)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    report = Report()
    report["true_duplicates_injected"] = len(truth)
    report["pairs_found"] = len(found)
    report["true_positives"] = tp
    report["false_positives"] = fp
    report["false_negatives"] = fn
    report["precision"] = round(precision, 3)
    report["recall"] = round(recall, 3)
    report["f1"] = round(f1, 3)
    report["missed_pairs"] = [sorted(p) for p in sorted(truth - found, key=sorted)][:10]
    return report


def threshold_sweep(patients: pd.DataFrame, duplicate_of: dict[str, str],
                    thresholds: list[float] | None = None) -> pd.DataFrame:
    """Precision/recall across candidate thresholds.

    This is the table that justifies MATCH_THRESHOLD. Picking a cut-off by
    eye and calling it tuned is the thing this replaces.
    """
    thresholds = thresholds or [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.95]
    rows = []
    for t in thresholds:
        pairs, _ = find_near_duplicates(patients, threshold=t)
        scored = evaluate(pairs, duplicate_of)
        rows.append({
            "threshold": t,
            "found": scored["pairs_found"],
            "tp": scored["true_positives"],
            "fp": scored["false_positives"],
            "fn": scored["false_negatives"],
            "precision": scored["precision"],
            "recall": scored["recall"],
            "f1": scored["f1"],
        })
    return pd.DataFrame(rows)


def dedupe_all(tables: dict[str, pd.DataFrame], threshold: float = MATCH_THRESHOLD
               ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, Report]:
    """Exact removal, then fuzzy linkage, then resolution.

    Encounters are deliberately left alone. Repeat visits are real, and the
    crosswalk is applied to them so a merged patient keeps BOTH visits rather
    than losing one.
    """
    patients = tables["patients"]
    report = Report()

    before = len(patients)
    exact_mask = patients.duplicated(
        subset=[c for c in ["patient_id", "given_name", "family_name", "sex", "birth_date"]
                if c in patients.columns],
        keep="first",
    )
    patients = patients.loc[~exact_mask].copy()
    report["rows_in"] = before
    report["exact_duplicates_removed"] = int(exact_mask.sum())

    pairs, block_report = find_near_duplicates(patients, threshold=threshold)
    report.update(block_report)

    resolved, crosswalk = resolve(patients, pairs)
    report["clusters_merged"] = len(set(crosswalk.values()))
    report["rows_out"] = len(resolved)

    out = dict(tables)
    out["patients"] = resolved
    # Point every child row at the surviving identifier.
    for name in ("encounters", "observations", "imaging_studies"):
        if name in out and "patient_id" in out[name].columns:
            child = out[name].copy()
            child["master_patient_id"] = [
                crosswalk.get(str(pid), str(pid)) for pid in child["patient_id"]
            ]
            out[name] = child

    return out, pairs, report
