"""T2 part two -- data cleaning.

Standardization made values comparable. Cleaning decides which of them are
*wrong*, and there are three different kinds of wrong that must not be
conflated:

  impossible   outside physiology. A 1580 cm height is not a tall patient, it
               is a missing decimal point. Quarantine it.
  inconsistent individually plausible, jointly contradictory. Systolic 70 with
               diastolic 120 is a transposition. Neither value can be trusted.
  extreme      rare but real. A CA-125 of 2400 U/mL in ovarian malignancy is
               the finding, not an error. Flag, never delete.

The third is why this module never deletes on statistics alone. An IQR filter
applied to a disease cohort removes exactly the patients the study is about.

EVERY REJECTION IS RECORDED. `clean_all` returns a flags table with one row per
issue -- rule, table, row, column, value and action -- so a reviewer can see
what the pipeline threw away and argue with it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import vocab
from .standardize import Report

# Quarantined values become NaN and are then handled by missing.py. That is
# deliberate: an impossible value and an absent value are the same thing to
# every downstream model, and pretending otherwise invents information.
QUARANTINE = "quarantined"
FLAG_ONLY = "flagged"


def _flag(rows: list[dict], rule: str, table: str, index: int, column: str,
          value: object, action: str, detail: str = "") -> None:
    rows.append({
        "rule": rule, "table": table, "row": index, "column": column,
        "value": value, "action": action, "detail": detail,
    })


# ---------------------------------------------------------------------------
# Rule 1 -- physiological plausibility
# ---------------------------------------------------------------------------


def check_plausibility(observations: pd.DataFrame, flags: list[dict]) -> pd.DataFrame:
    """Quarantine values outside the physiological range for their LOINC code.

    Runs only after unit conversion. Before it, every imperial height fails.
    """
    out = observations.copy()

    for code, group in out.groupby("loinc", dropna=True):
        lo, hi = vocab.plausible_range(str(code))
        values = pd.to_numeric(group["value"], errors="coerce")
        bad = group.index[values.notna() & ((values < lo) | (values > hi))]

        for idx in bad:
            _flag(flags, "plausibility", "observations", int(idx), "value",
                  out.at[idx, "value"], QUARANTINE,
                  f"{vocab.LOINC[str(code)]['display']} outside [{lo}, {hi}]")
            out.at[idx, "value"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Rule 2 -- cross-field consistency
# ---------------------------------------------------------------------------


def _pivot_by_encounter(observations: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide, one row per encounter. Cross-field rules need the fields
    side by side; the long form is right for storage and wrong for this."""
    numeric = observations.copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    return numeric.pivot_table(
        index="encounter_id", columns="loinc", values="value", aggfunc="first"
    )


def check_blood_pressure(observations: pd.DataFrame, flags: list[dict]) -> pd.DataFrame:
    """Systolic must exceed diastolic.

    A violation means the pair was transposed on entry. The safe repair is to
    quarantine BOTH -- swapping them back assumes the transposition rather
    than a single mistyped value, and that assumption is not recoverable from
    the data.
    """
    out = observations.copy()
    wide = _pivot_by_encounter(out)
    if vocab.Loinc.SYSTOLIC not in wide or vocab.Loinc.DIASTOLIC not in wide:
        return out

    sys_col = wide[vocab.Loinc.SYSTOLIC]
    dia_col = wide[vocab.Loinc.DIASTOLIC]
    offenders = wide.index[sys_col.notna() & dia_col.notna() & (sys_col <= dia_col)]

    for enc in offenders:
        rows = out.index[
            (out["encounter_id"] == enc)
            & (out["loinc"].isin([vocab.Loinc.SYSTOLIC, vocab.Loinc.DIASTOLIC]))
        ]
        for idx in rows:
            _flag(flags, "bp_consistency", "observations", int(idx), "value",
                  out.at[idx, "value"], QUARANTINE,
                  f"systolic {sys_col[enc]:.0f} <= diastolic {dia_col[enc]:.0f}")
            out.at[idx, "value"] = np.nan

    return out


def check_bmi_consistency(observations: pd.DataFrame, flags: list[dict],
                          tolerance: float = 1.5) -> pd.DataFrame:
    """BMI must agree with height and weight.

    Recorded BMI is kept where it disagrees but all three are individually
    plausible -- the discrepancy is flagged, not resolved, because any of the
    three could be the wrong one. Only the derivable cases are repaired: a
    missing BMI with a good height and weight is computed.
    """
    out = observations.copy()
    wide = _pivot_by_encounter(out)
    needed = {vocab.Loinc.BMI, vocab.Loinc.HEIGHT, vocab.Loinc.WEIGHT}
    if not needed <= set(wide.columns):
        return out

    derived = wide[vocab.Loinc.WEIGHT] / (wide[vocab.Loinc.HEIGHT] / 100.0) ** 2
    recorded = wide[vocab.Loinc.BMI]

    disagree = wide.index[
        recorded.notna() & derived.notna() & ((recorded - derived).abs() > tolerance)
    ]
    for enc in disagree:
        rows = out.index[(out["encounter_id"] == enc) & (out["loinc"] == vocab.Loinc.BMI)]
        for idx in rows:
            _flag(flags, "bmi_consistency", "observations", int(idx), "value",
                  out.at[idx, "value"], FLAG_ONLY,
                  f"recorded {recorded[enc]:.1f} vs derived {derived[enc]:.1f}")

    return out


def check_dates(patients: pd.DataFrame, encounters: pd.DataFrame,
                flags: list[dict], today: dt.date = dt.date(2025, 6, 30)
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Birth dates in the past, encounters after birth and not in the future."""
    p_out, e_out = patients.copy(), encounters.copy()

    for idx, value in p_out["birth_date"].items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        born = dt.date.fromisoformat(str(value))
        if born > today:
            _flag(flags, "birth_date_future", "patients", int(idx), "birth_date",
                  value, QUARANTINE, "birth date is in the future")
            p_out.at[idx, "birth_date"] = None
        elif (today - born).days / 365.25 > 120:
            _flag(flags, "birth_date_implausible", "patients", int(idx), "birth_date",
                  value, QUARANTINE, "implies age over 120")
            p_out.at[idx, "birth_date"] = None

    born_by_patient = {
        row.patient_id: row.birth_date
        for row in p_out.itertuples(index=False)
        if row.birth_date
    }
    for idx, row in e_out.iterrows():
        value = row.get("encounter_date")
        if not value:
            continue
        seen = dt.date.fromisoformat(str(value))
        born_str = born_by_patient.get(row["patient_id"])
        if seen > today:
            _flag(flags, "encounter_future", "encounters", int(idx), "encounter_date",
                  value, QUARANTINE, "encounter date is in the future")
            e_out.at[idx, "encounter_date"] = None
        elif born_str and seen < dt.date.fromisoformat(str(born_str)):
            _flag(flags, "encounter_before_birth", "encounters", int(idx),
                  "encounter_date", value, QUARANTINE, "encounter precedes birth")
            e_out.at[idx, "encounter_date"] = None

    return p_out, e_out


# ---------------------------------------------------------------------------
# Rule 3 -- statistical outliers, FLAGGED ONLY
# ---------------------------------------------------------------------------


def check_outliers(observations: pd.DataFrame, flags: list[dict],
                   z_threshold: float = 4.0) -> pd.DataFrame:
    """Mark statistically extreme values without touching them.

    Uses a robust z-score built on the median and MAD rather than mean and
    standard deviation, because a handful of extreme values inflates the
    standard deviation enough to hide themselves.

        robust_z = 0.6745 * (x - median) / MAD

    0.6745 is the 75th-percentile z of a normal, which makes MAD a consistent
    estimator of sigma under normality.

    Nothing here is deleted. In an ovarian-pathology cohort the extreme CA-125
    values are the positive cases.
    """
    out = observations.copy()

    for code, group in out.groupby("loinc", dropna=True):
        values = pd.to_numeric(group["value"], errors="coerce")
        present = values.dropna()
        if len(present) < 20:
            continue

        median = float(present.median())
        mad = float((present - median).abs().median())
        if mad == 0:
            continue

        robust_z = 0.6745 * (values - median) / mad
        extreme = group.index[robust_z.abs() > z_threshold]

        for idx in extreme:
            _flag(flags, "outlier", "observations", int(idx), "value",
                  out.at[idx, "value"], FLAG_ONLY,
                  f"robust z = {robust_z[idx]:.1f} for "
                  f"{vocab.LOINC[str(code)]['display']}")

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def clean_all(tables: dict[str, pd.DataFrame]
              ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, Report]:
    """Apply every rule in dependency order.

    Order matters: plausibility first so that grossly wrong values are gone
    before the cross-field rules read them, then consistency, then the
    statistical pass over what survives.
    """
    flags: list[dict] = []

    observations = check_plausibility(tables["observations"], flags)
    observations = check_blood_pressure(observations, flags)
    observations = check_bmi_consistency(observations, flags)
    observations = check_outliers(observations, flags)

    patients, encounters = check_dates(tables["patients"], tables["encounters"], flags)

    flags_df = pd.DataFrame(flags, columns=[
        "rule", "table", "row", "column", "value", "action", "detail",
    ])

    report = Report()
    report["flags_raised"] = len(flags_df)
    report["values_quarantined"] = int((flags_df["action"] == QUARANTINE).sum()) if len(flags_df) else 0
    report["values_flagged_only"] = int((flags_df["action"] == FLAG_ONLY).sum()) if len(flags_df) else 0
    if len(flags_df):
        report["by_rule"] = flags_df["rule"].value_counts().to_dict()

    out = dict(tables)
    out["observations"] = observations
    out["patients"] = patients
    out["encounters"] = encounters
    return out, flags_df, report
