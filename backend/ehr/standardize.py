"""T2 part one -- standardization.

Standardization is the step that makes records from different sources
*comparable*. It changes representation, never meaning:

  syntactic   "29/06/2007", "06-29-2007" and "29 Jun 2007" are one date
  unit        58.65 inches and 148.97 cm are one height
  semantic    "Hb", "HGB" and "Haemoglobin" are one LOINC concept
  categorical "F", "woman" and "2" are one administrative gender

Cleaning (clean.py) is the different step that decides a value is *wrong*.
Doing them in the wrong order is the classic mistake: a range check applied
before unit conversion rejects every height recorded at the imperial site,
because 58.65 looks impossible next to a 40-230 cm rule.

Every function here returns the transformed data plus a count of what it could
not handle. Silent coercion is how a pipeline loses a third of a cohort
without anyone noticing.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import numpy as np
import pandas as pd

from . import vocab

# Accepted date layouts, most specific first. %d/%m vs %m/%d is genuinely
# ambiguous for day <= 12; see parse_date.
_DATE_PATTERNS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "%d %B %Y",
    "%Y/%m/%d", "%d-%m-%Y", "%b %d %Y", "%d.%m.%Y",
]

# A number, optionally with a comma decimal, optionally followed by a unit.
_NUMERIC_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*([A-Za-z/%^*\[\]0-9]*)\s*$")


class Report(dict):
    """A dict that prints as a readable block. Every transform returns one, so
    a run can be audited without a debugger."""

    def __str__(self) -> str:
        width = max((len(k) for k in self), default=0)
        return "\n".join(f"  {k:<{width}}  {v}" for k, v in self.items())


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def is_null_token(value: Any) -> bool:
    """True for anything meaning no-value-recorded."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return vocab.normalise_key(value) in vocab.NULL_TOKENS


def parse_numeric(value: Any) -> tuple[float | None, str | None]:
    """Return (number, embedded_unit).

    Handles the three ways a number arrives non-numeric in an extract: a
    comma decimal, a unit glued on the end, and a null token in a numeric
    column. Returns (None, None) when there is no number to find.
    """
    if is_null_token(value):
        return None, None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), None

    match = _NUMERIC_RE.match(str(value))
    if not match:
        return None, None

    number, unit = match.group(1), match.group(2)
    return float(number.replace(",", ".")), (unit or None)


def parse_date(value: Any, dayfirst_default: bool = True) -> str | None:
    """Parse a date written in any of the layouts sites actually use, and
    return it as an ISO-8601 string.

    AMBIGUITY
        03/04/1990 is 3 April under %d/%m and 4 March under %m/%d, and nothing
        in the string resolves it. The generator here emits %d/%m for one site
        and %m-%d for another, so the separator happens to disambiguate -- but
        that is a property of this cohort, not a general rule. Real intake
        needs the source system documented. dayfirst_default records which
        assumption was made rather than hiding it.
    """
    if is_null_token(value):
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.date().isoformat() if isinstance(value, dt.datetime) else value.isoformat()

    text = " ".join(str(value).strip().split())
    patterns = _DATE_PATTERNS if dayfirst_default else (
        ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"] + _DATE_PATTERNS
    )
    for pattern in patterns:
        try:
            return dt.datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def standardise_unit(unit: Any) -> str | None:
    """Map a written unit onto its canonical spelling (UCUM-flavoured)."""
    if is_null_token(unit):
        return None
    key = vocab.normalise_key(unit)
    return vocab.UNIT_ALIASES.get(key, str(unit).strip())


def convert_value(value: float, from_unit: str | None, to_unit: str) -> float | None:
    """Convert onto the canonical unit. None when no conversion is known --
    the caller must treat that as unmapped, not as a pass-through."""
    if from_unit is None or from_unit == to_unit:
        return value

    # Temperature is affine, so it cannot live in the multiplier table.
    if from_unit == "[degF]" and to_unit == "Cel":
        return (value - 32.0) * 5.0 / 9.0
    if from_unit == "Cel" and to_unit == "[degF]":
        return value * 9.0 / 5.0 + 32.0

    factor = vocab.UNIT_FACTORS.get((from_unit, to_unit))
    if factor is not None:
        return value * factor

    inverse = vocab.UNIT_FACTORS.get((to_unit, from_unit))
    if inverse:
        return value / inverse
    return None


def standardise_sex(value: Any) -> str:
    """Free-text sex onto the FHIR administrative-gender value set."""
    return vocab.SEX_SYNONYMS.get(vocab.normalise_key(value), "unknown")


def standardise_name(value: Any) -> str:
    """Trim, collapse spacing, title-case.

    Title-casing is lossy for names like `McDonald` and `van der Berg`, which
    is a real limitation -- it is applied here only because the matcher in
    dedupe.py needs a stable form to compare, and it compares case-folded
    anyway. The original is preserved in the *_raw columns.
    """
    if is_null_token(value):
        return ""
    return " ".join(str(value).strip().split()).title()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def standardise_patients(patients: pd.DataFrame) -> tuple[pd.DataFrame, Report]:
    """Names, sex and birth dates onto canonical forms.

    Raw values are kept in *_raw columns. An EHR transform that destroys the
    source value cannot be audited, and provenance is the whole argument for
    doing this in a pipeline rather than by hand.
    """
    out = patients.copy()
    report = Report()

    out["given_name_raw"] = out["given_name"]
    out["family_name_raw"] = out["family_name"]
    out["sex_raw"] = out["sex"]
    out["birth_date_raw"] = out["birth_date"]

    out["given_name"] = out["given_name"].map(standardise_name)
    out["family_name"] = out["family_name"].map(standardise_name)
    out["sex"] = out["sex"].map(standardise_sex)
    out["birth_date"] = out["birth_date"].map(parse_date)

    unparsed = int(out["birth_date"].isna().sum())
    report["rows"] = len(out)
    report["birth_dates_unparsed"] = unparsed
    report["sex_unknown_after_mapping"] = int((out["sex"] == "unknown").sum())
    report["sex_distinct_before"] = out["sex_raw"].nunique()
    report["sex_distinct_after"] = out["sex"].nunique()

    # Age is derived, never stored: a stored age is wrong the day after intake.
    today = dt.date(2025, 6, 30)
    out["age_years"] = [
        None if d is None else round((today - dt.date.fromisoformat(d)).days / 365.25, 1)
        for d in out["birth_date"]
    ]
    return out, report


def standardise_encounters(encounters: pd.DataFrame) -> tuple[pd.DataFrame, Report]:
    out = encounters.copy()
    out["encounter_date_raw"] = out["encounter_date"]
    out["encounter_date"] = out["encounter_date"].map(parse_date)

    report = Report()
    report["rows"] = len(out)
    report["dates_unparsed"] = int(out["encounter_date"].isna().sum())
    return out, report


def standardise_observations(observations: pd.DataFrame) -> tuple[pd.DataFrame, Report]:
    """The core mapping: local test name -> LOINC, local unit -> canonical unit.

    Produces one row per observation with `loinc`, `value` (canonical unit) and
    `value_raw`/`unit_raw` preserved. Rows whose test name has no LOINC mapping
    are kept with loinc=None and counted -- dropping them would silently shrink
    the cohort.
    """
    out = observations.copy()
    report = Report()

    out["local_test_raw"] = out["local_test_name"]
    out["value_raw"] = out["value"]
    out["unit_raw"] = out["unit"]

    loincs, values, units = [], [], []
    unmapped_tests: set[str] = set()
    unconvertible: set[tuple[str, str]] = set()
    embedded_unit_used = 0

    for row in out.itertuples(index=False):
        code = vocab.LOCAL_TEST_TO_LOINC.get(vocab.normalise_key(row.local_test_name))
        loincs.append(code)

        if code is None:
            unmapped_tests.add(str(row.local_test_name))
            values.append(np.nan)
            units.append(None)
            continue

        number, embedded = parse_numeric(row.value)
        if number is None:
            values.append(np.nan)          # null token, or unparseable text
            units.append(vocab.canonical_unit(code))
            continue

        # A unit glued to the value overrides the unit column: it was typed by
        # the same person, at the same moment, as the number.
        source_unit = standardise_unit(embedded or row.unit)
        if embedded:
            embedded_unit_used += 1

        target = vocab.canonical_unit(code)
        converted = convert_value(number, source_unit, target)
        if converted is None:
            unconvertible.add((str(source_unit), target))
            values.append(np.nan)
        else:
            values.append(round(converted, 3))
        units.append(target)

    out["loinc"] = loincs
    out["value"] = values
    out["unit"] = units
    out["display"] = [None if c is None else vocab.LOINC[c]["display"] for c in loincs]

    report["rows"] = len(out)
    report["distinct_local_test_names"] = out["local_test_raw"].nunique()
    report["mapped_to_loinc"] = int(out["loinc"].notna().sum())
    report["unmapped_test_names"] = sorted(unmapped_tests) or "none"
    report["distinct_units_before"] = out["unit_raw"].nunique()
    report["distinct_units_after"] = int(out["unit"].nunique())
    report["values_with_embedded_unit"] = embedded_unit_used
    report["unconvertible_unit_pairs"] = sorted(unconvertible) or "none"
    report["values_null_after_parse"] = int(out["value"].isna().sum())
    return out, report


def standardise_diagnoses(studies: pd.DataFrame) -> tuple[pd.DataFrame, Report]:
    """Free-text clinician diagnosis -> ICD-10.

    Unmapped text is kept with icd10=None rather than guessed. An incorrectly
    coded diagnosis is worse than an uncoded one: it is wrong and it looks
    complete.
    """
    out = studies.copy()
    report = Report()

    out["local_diagnosis_raw"] = out["local_diagnosis"]
    codes = [
        vocab.DIAGNOSIS_SYNONYMS.get(vocab.normalise_key(text))
        for text in out["local_diagnosis"]
    ]
    out["icd10"] = codes
    out["icd10_display"] = [None if c is None else vocab.ICD10[c] for c in codes]

    unmapped = sorted({
        str(t) for t, c in zip(out["local_diagnosis"], codes) if c is None
    })
    report["rows"] = len(out)
    report["distinct_diagnosis_text"] = out["local_diagnosis_raw"].nunique()
    report["mapped_to_icd10"] = int(out["icd10"].notna().sum())
    report["distinct_icd10_codes"] = int(out["icd10"].nunique())
    report["unmapped_diagnosis_text"] = unmapped or "none"
    return out, report


def standardise_all(tables: dict[str, pd.DataFrame]
                    ) -> tuple[dict[str, pd.DataFrame], dict[str, Report]]:
    """Run every standardiser. Returns new tables plus one report per table."""
    patients, rep_p = standardise_patients(tables["patients"])
    encounters, rep_e = standardise_encounters(tables["encounters"])
    observations, rep_o = standardise_observations(tables["observations"])
    studies, rep_s = standardise_diagnoses(tables["imaging_studies"])

    return (
        {
            "patients": patients,
            "encounters": encounters,
            "observations": observations,
            "imaging_studies": studies,
        },
        {
            "patients": rep_p,
            "encounters": rep_e,
            "observations": rep_o,
            "imaging_studies": rep_s,
        },
    )
