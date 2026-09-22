"""Synthetic EHR cohort keyed to MMOTU scan ids -- the data source for T1-T3.

WHY SYNTHETIC
    T2 and T3 are about *recovering* from defects. Judging a repair needs the
    original, and no real de-identified extract ships one. So this module
    builds a clean cohort first, keeps it as ground truth, then corrupts a copy
    in ways whose exact extent is recorded. Every downstream claim -- "fuzzy
    linkage found 47 of 50 duplicates", "KNN imputation beat the median by
    0.8 BMI points" -- is then measurable rather than asserted.

    Everything here is fabricated. No real patient data is involved.

WHY KEYED TO MMOTU
    Each synthetic patient owns one ultrasound from the MMOTU OTU_2d set, so
    the EHR half (Unit-1) and the imaging half (Unit-2) describe the same
    cohort instead of being two unrelated demos. A scan stem in
    imaging_studies.scan_stem is the join key back to the image files that
    data_prep/mmotu_to_yolo.py converts.

OUTPUT
    Four tables, in the shape a real extract arrives in:

      patients          one row per registration event (NOT per person --
                        that is the T3 problem)
      encounters        one row per visit
      observations      LONG format: one row per measurement, with the local
                        test name and unit as typed
      imaging_studies   one row per scan, carrying the local diagnosis label

    build() returns a Cohort holding the clean tables, the dirty tables and a
    Manifest recording precisely what was corrupted.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
from typing import Any

import numpy as np
import pandas as pd

from . import vocab

# ---------------------------------------------------------------------------
# Name pools. Small on purpose: collisions are realistic and they give the
# fuzzy matcher in dedupe.py genuine false-positive pressure to handle.
# ---------------------------------------------------------------------------

_GIVEN = [
    "Aditi", "Priya", "Meera", "Kavya", "Ananya", "Divya", "Lakshmi", "Nandini",
    "Sneha", "Pooja", "Rekha", "Swathi", "Harini", "Deepa", "Anjali", "Ritika",
    "Shreya", "Vaishnavi", "Padma", "Geetha", "Sushmita", "Malini", "Indira",
    "Radhika", "Chitra", "Bhavani", "Yamini", "Tanvi", "Keerthi", "Sowmya",
]

_FAMILY = [
    "Iyer", "Menon", "Nair", "Reddy", "Rao", "Sharma", "Verma", "Patel",
    "Krishnan", "Subramanian", "Pillai", "Desai", "Joshi", "Kulkarni", "Naidu",
    "Chandran", "Bhat", "Shetty", "Gupta", "Mehta",
]

# Nickname / transliteration variants used when forging a duplicate record.
_NAME_VARIANTS: dict[str, str] = {
    "Aditi": "Adithi", "Priya": "Prya", "Meera": "Mira", "Kavya": "Kavyaa",
    "Ananya": "Anannya", "Divya": "Dhivya", "Lakshmi": "Laxmi",
    "Nandini": "Nandhini", "Sneha": "Snehaa", "Pooja": "Puja",
    "Swathi": "Swathy", "Harini": "Harinee", "Deepa": "Dipa",
    "Shreya": "Shriya", "Geetha": "Gita", "Sowmya": "Soumya",
    "Iyer": "Ayyar", "Menon": "Menonn", "Nair": "Nayar", "Reddy": "Reddi",
    "Krishnan": "Krishnaa", "Subramanian": "Subramaniam", "Pillai": "Pilla",
    "Kulkarni": "Kulkarny", "Chandran": "Chandra", "Shetty": "Setty",
}

_SITES = ["RGH", "CMC", "APX"]          # three feeding hospitals
_DEPARTMENTS = ["Gynaecology", "Radiology", "Emergency", "Oncology"]
_SYMPTOMS = list(vocab.SNOMED_FINDINGS)

# Which observations every encounter carries, and how each site spells them.
# One concept, three local names -- that is the whole of the T2 mapping job.
_LOCAL_TEST_NAMES: dict[str, list[str]] = {
    vocab.Loinc.HEIGHT: ["Height", "HT", "stature"],
    vocab.Loinc.WEIGHT: ["Weight", "WT", "mass"],
    vocab.Loinc.SYSTOLIC: ["SBP", "Systolic BP", "BP systolic"],
    vocab.Loinc.DIASTOLIC: ["DBP", "Diastolic BP", "BP diastolic"],
    vocab.Loinc.HEART_RATE: ["HR", "Pulse", "heart rate"],
    vocab.Loinc.TEMPERATURE: ["Temp", "Temperature", "body temperature"],
    vocab.Loinc.HEMOGLOBIN: ["Hb", "HGB", "Haemoglobin"],
    vocab.Loinc.LEUKOCYTES: ["WBC", "TLC", "white cell count"],
    vocab.Loinc.CREATININE: ["Creat", "Creatinine", "S Creatinine"],
    vocab.Loinc.CA125: ["CA125", "CA-125", "Cancer Antigen 125"],
    vocab.Loinc.BMI: ["BMI", "Body Mass Index", "Quetelet index"],
}

# Per-site unit conventions. Site 0 is metric-canonical, site 1 is imperial for
# anthropometry, site 2 uses SI lab units -- which is why unit conversion has
# to happen before any cross-site comparison.
_SITE_UNITS: dict[str, dict[str, str]] = {
    "RGH": {
        vocab.Loinc.HEIGHT: "cm", vocab.Loinc.WEIGHT: "kg",
        vocab.Loinc.TEMPERATURE: "C", vocab.Loinc.HEMOGLOBIN: "g/dL",
        vocab.Loinc.LEUKOCYTES: "10^9/L", vocab.Loinc.CREATININE: "mg/dL",
        vocab.Loinc.CA125: "U/mL",
    },
    "CMC": {
        vocab.Loinc.HEIGHT: "inches", vocab.Loinc.WEIGHT: "lbs",
        vocab.Loinc.TEMPERATURE: "F", vocab.Loinc.HEMOGLOBIN: "g/dL",
        vocab.Loinc.LEUKOCYTES: "/uL", vocab.Loinc.CREATININE: "mg/dL",
        vocab.Loinc.CA125: "U/mL",
    },
    "APX": {
        vocab.Loinc.HEIGHT: "m", vocab.Loinc.WEIGHT: "kg",
        vocab.Loinc.TEMPERATURE: "C", vocab.Loinc.HEMOGLOBIN: "g/L",
        vocab.Loinc.LEUKOCYTES: "10^9/L", vocab.Loinc.CREATININE: "umol/L",
        vocab.Loinc.CA125: "kU/L",
    },
}

# Date formats, one per site plus a long form some clerks prefer.
_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y"]

# Sex is recorded as free text at registration. The cohort is gynaecological,
# so the truth is always female; the spelling is not.
_SEX_RAW = ["F", "f", "Female", "FEMALE", "female", "W", "woman", "2"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Manifest:
    """Exactly what was corrupted. The answer key for every T2/T3 metric."""

    # (table, row_index, column) triples whose value was replaced by a null token
    nulled: list[tuple[str, int, str]] = dataclasses.field(default_factory=list)
    # missingness mechanism per observation concept
    missing_mechanism: dict[str, str] = dataclasses.field(default_factory=dict)
    # forged duplicate patient_id -> the patient_id it duplicates
    duplicate_of: dict[str, str] = dataclasses.field(default_factory=dict)
    # patient_ids that are exact row-level duplicates
    exact_duplicate_ids: list[str] = dataclasses.field(default_factory=list)
    # (table, row_index, column) whose value was made physiologically impossible
    implausible: list[tuple[str, int, str]] = dataclasses.field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "cells_nulled": len(self.nulled),
            "forged_duplicates": len(self.duplicate_of),
            "exact_duplicates": len(self.exact_duplicate_ids),
            "implausible_values": len(self.implausible),
        }


@dataclasses.dataclass
class Cohort:
    clean: dict[str, pd.DataFrame]
    dirty: dict[str, pd.DataFrame]
    manifest: Manifest

    def write(self, out_dir: str) -> str:
        """Write both versions to CSV. `dirty` is what the pipeline consumes;
        `clean` is held back as ground truth."""
        for label, tables in (("clean", self.clean), ("raw", self.dirty)):
            d = os.path.join(out_dir, label)
            os.makedirs(d, exist_ok=True)
            for name, df in tables.items():
                df.to_csv(os.path.join(d, f"{name}.csv"), index=False)
        return out_dir


# ---------------------------------------------------------------------------
# Clean generation
# ---------------------------------------------------------------------------


def _build_clean(rng: np.random.Generator, n: int, scan_ids: list[int]
                 ) -> dict[str, pd.DataFrame]:
    """One patient, one encounter, one scan, eleven observations each.

    Values are drawn from plausible distributions and are internally
    consistent: BMI is computed from height and weight rather than sampled, so
    clean.py has a real cross-field rule to check.
    """
    today = dt.date(2025, 6, 30)

    patients, encounters, studies, observations = [], [], [], []

    for i in range(n):
        pid = f"MRN{100000 + i}"
        given = str(rng.choice(_GIVEN))
        family = str(rng.choice(_FAMILY))
        site = _SITES[i % len(_SITES)]

        # Ovarian pathology skews to reproductive age; allow a post-menopausal tail.
        age = int(np.clip(rng.normal(38, 13), 16, 88))
        dob = today - dt.timedelta(days=age * 365 + int(rng.integers(0, 365)))

        patients.append({
            "patient_id": pid,
            "given_name": given,
            "family_name": family,
            "sex": "female",
            "birth_date": dob.isoformat(),
            "site": site,
        })

        enc_id = f"ENC{200000 + i}"
        enc_date = today - dt.timedelta(days=int(rng.integers(0, 720)))
        encounters.append({
            "encounter_id": enc_id,
            "patient_id": pid,
            "encounter_date": enc_date.isoformat(),
            "department": str(rng.choice(_DEPARTMENTS)),
            "presenting_symptom": str(rng.choice(_SYMPTOMS)),
        })

        # Imaging study -- the join back to the MMOTU image files.
        klass = int(rng.integers(0, 8))
        local_label, icd10 = vocab.MMOTU_CLASS_TO_ICD10[klass]
        studies.append({
            "study_id": f"IMG{300000 + i}",
            "encounter_id": enc_id,
            "patient_id": pid,
            "scan_stem": str(scan_ids[i % len(scan_ids)]),
            "modality": "US",
            "mmotu_class": klass,
            "local_diagnosis": local_label,
            "icd10": icd10,
        })

        # --- observations, in canonical units -----------------------------
        height = float(np.clip(rng.normal(158, 7), 138, 182))
        weight = float(np.clip(rng.normal(62, 12), 36, 130))
        bmi = weight / (height / 100.0) ** 2
        systolic = float(np.clip(rng.normal(118, 13), 85, 185))
        # Diastolic is derived so systolic > diastolic always holds in the
        # clean data; clean.py checks that the dirty data still does.
        diastolic = float(np.clip(systolic - rng.normal(40, 7), 50, 110))

        # CA-125 is log-normal and genuinely elevated in malignancy. Class 7
        # (high grade serous) is the one that should separate.
        ca125_base = 4.2 if klass == 7 else 2.6
        ca125 = float(np.exp(rng.normal(ca125_base, 0.55)))

        values = {
            vocab.Loinc.HEIGHT: height,
            vocab.Loinc.WEIGHT: weight,
            vocab.Loinc.BMI: bmi,
            vocab.Loinc.SYSTOLIC: systolic,
            vocab.Loinc.DIASTOLIC: diastolic,
            vocab.Loinc.HEART_RATE: float(np.clip(rng.normal(76, 11), 48, 130)),
            vocab.Loinc.TEMPERATURE: float(np.clip(rng.normal(36.8, 0.4), 35.4, 39.5)),
            vocab.Loinc.HEMOGLOBIN: float(np.clip(rng.normal(12.4, 1.5), 7.0, 17.0)),
            vocab.Loinc.LEUKOCYTES: float(np.clip(rng.normal(7.4, 2.1), 2.5, 18.0)),
            vocab.Loinc.CREATININE: float(np.clip(rng.normal(0.78, 0.18), 0.35, 2.2)),
            vocab.Loinc.CA125: ca125,
        }

        for code, value in values.items():
            observations.append({
                "observation_id": f"OBS{len(observations) + 400000}",
                "encounter_id": enc_id,
                "patient_id": pid,
                "loinc": code,
                "display": vocab.LOINC[code]["display"],
                "value": round(value, 2),
                "unit": vocab.canonical_unit(code),
            })

    return {
        "patients": pd.DataFrame(patients),
        "encounters": pd.DataFrame(encounters),
        "imaging_studies": pd.DataFrame(studies),
        "observations": pd.DataFrame(observations),
    }


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------


def _to_local_units(value: float, code: str, site: str,
                    rng: np.random.Generator) -> tuple[Any, str]:
    """Express a canonical value in the site local unit, then render it the way
    a human types it -- sometimes with a comma decimal or a trailing unit."""
    unit = _SITE_UNITS[site].get(code)
    if unit is None:
        return round(value, 2), vocab.canonical_unit(code)

    converted = value
    if code == vocab.Loinc.HEIGHT:
        converted = {"cm": value, "m": value / 100.0, "inches": value / 2.54}[unit]
    elif code == vocab.Loinc.WEIGHT:
        converted = value / 0.45359237 if unit == "lbs" else value
    elif code == vocab.Loinc.TEMPERATURE:
        converted = value * 9 / 5 + 32 if unit == "F" else value
    elif code == vocab.Loinc.HEMOGLOBIN:
        converted = value * 10.0 if unit == "g/L" else value
    elif code == vocab.Loinc.LEUKOCYTES:
        converted = value * 1000.0 if unit == "/uL" else value
    elif code == vocab.Loinc.CREATININE:
        converted = value / 0.0113122 if unit == "umol/L" else value

    rendered: Any = round(converted, 2)

    # Textual noise: a European decimal comma, or the unit glued to the number.
    roll = rng.random()
    if roll < 0.04:
        rendered = str(rendered).replace(".", ",")
    elif roll < 0.07:
        rendered = f"{rendered} {unit}"

    return rendered, unit


def _corrupt(clean: dict[str, pd.DataFrame], rng: np.random.Generator,
             manifest: Manifest) -> dict[str, pd.DataFrame]:
    """Turn the clean cohort into something that looks like a real extract."""
    patients = clean["patients"].copy()
    encounters = clean["encounters"].copy()
    studies = clean["imaging_studies"].copy()
    observations = clean["observations"].copy()

    site_of = dict(zip(patients["patient_id"], patients["site"]))

    # --- T2: patients -----------------------------------------------------
    # Names arrive with stray case and whitespace; sex as free text; dates in
    # whichever format the originating system used.
    def _messy_name(name: str) -> str:
        roll = rng.random()
        if roll < 0.12:
            return name.upper()
        if roll < 0.22:
            return name.lower()
        if roll < 0.30:
            return f"  {name} "
        return name

    patients["given_name"] = [_messy_name(n) for n in patients["given_name"]]
    patients["family_name"] = [_messy_name(n) for n in patients["family_name"]]
    patients["sex"] = [str(rng.choice(_SEX_RAW)) for _ in range(len(patients))]
    patients["birth_date"] = [
        dt.date.fromisoformat(d).strftime(_DATE_FORMATS[i % len(_DATE_FORMATS)])
        for i, d in enumerate(patients["birth_date"])
    ]
    encounters["encounter_date"] = [
        dt.date.fromisoformat(d).strftime(_DATE_FORMATS[(i + 1) % len(_DATE_FORMATS)])
        for i, d in enumerate(encounters["encounter_date"])
    ]

    # --- T2: observations -------------------------------------------------
    # Local test names and site units replace the canonical pair.
    local_names, raw_values, raw_units = [], [], []
    for row in observations.itertuples(index=False):
        code = row.loinc
        site = site_of[row.patient_id]
        names = _LOCAL_TEST_NAMES[code]
        local_names.append(names[_SITES.index(site) % len(names)])
        value, unit = _to_local_units(float(row.value), code, site, rng)
        raw_values.append(value)
        raw_units.append(unit)

    observations = observations.drop(columns=["loinc", "display", "unit"])
    observations["local_test_name"] = local_names
    observations["value"] = raw_values
    observations["unit"] = raw_units

    # --- T2: diagnoses ----------------------------------------------------
    # Replace the tidy dataset label with a clinician synonym, and drop the
    # pre-coded icd10 column entirely -- deriving it is the exercise.
    by_icd: dict[str, list[str]] = {}
    for text, code in vocab.DIAGNOSIS_SYNONYMS.items():
        by_icd.setdefault(code, []).append(text)
    studies["local_diagnosis"] = [
        str(rng.choice(by_icd[code])) for code in studies["icd10"]
    ]
    studies = studies.drop(columns=["icd10", "mmotu_class"])

    # --- T2: implausible values ------------------------------------------
    # Transcription slips that produce impossible physiology. Separate from
    # missingness: these are present, wrong, and must be caught by range rules.
    impossible = {
        vocab.Loinc.HEIGHT: 1580.0,       # cm entered without the decimal
        vocab.Loinc.WEIGHT: 0.0,          # scale never read
        vocab.Loinc.HEART_RATE: 999.0,    # sentinel typed into a numeric field
        vocab.Loinc.CREATININE: -1.2,     # sign slip
        vocab.Loinc.TEMPERATURE: 3.68,    # decimal shifted
    }
    obs_codes = clean["observations"]["loinc"].to_numpy()
    for code, bad in impossible.items():
        candidates = np.flatnonzero(obs_codes == code)
        for idx in rng.choice(candidates, size=max(1, len(candidates) // 60), replace=False):
            observations.at[int(idx), "value"] = bad
            manifest.implausible.append(("observations", int(idx), "value"))

    # Systolic/diastolic swapped on a few rows: each value is individually
    # plausible, so only the cross-field rule catches it.
    sys_idx = np.flatnonzero(obs_codes == vocab.Loinc.SYSTOLIC)
    for idx in rng.choice(sys_idx, size=max(1, len(sys_idx) // 50), replace=False):
        enc = observations.at[int(idx), "encounter_id"]
        dia_rows = observations.index[
            (observations["encounter_id"] == enc)
            & (clean["observations"]["loinc"] == vocab.Loinc.DIASTOLIC)
        ]
        if len(dia_rows):
            j = int(dia_rows[0])
            a, b = observations.at[int(idx), "value"], observations.at[j, "value"]
            try:
                observations.at[int(idx), "value"], observations.at[j, "value"] = b, a
                manifest.implausible.append(("observations", int(idx), "value"))
            except (TypeError, ValueError):
                pass  # already a string-rendered value; leave it alone

    # --- T3: missingness, three mechanisms --------------------------------
    observations = _inject_missingness(observations, clean, patients, rng, manifest)

    # A few whole columns go sparse in the patient table too.
    for col, rate in (("site", 0.05),):
        victims = rng.choice(patients.index, size=int(len(patients) * rate), replace=False)
        for idx in victims:
            patients.at[int(idx), col] = ""
            manifest.nulled.append(("patients", int(idx), col))

    # --- T3: duplicates ---------------------------------------------------
    patients, encounters = _inject_duplicates(patients, encounters, rng, manifest)

    return {
        "patients": patients,
        "encounters": encounters,
        "imaging_studies": studies,
        "observations": observations,
    }


def _inject_missingness(observations: pd.DataFrame, clean: dict[str, pd.DataFrame],
                        patients: pd.DataFrame, rng: np.random.Generator,
                        manifest: Manifest) -> pd.DataFrame:
    """Delete values under three named mechanisms.

    The distinction decides whether imputation is honest:

      MCAR  missing at random, independent of everything. Dropping rows is
            unbiased, only wasteful.
      MAR   missingness depends on an OBSERVED variable. Imputation that
            conditions on that variable recovers the distribution.
      MNAR  missingness depends on the UNOBSERVED value itself. No imputation
            fixes this; it can only be modelled or declared.
    """
    codes = clean["observations"]["loinc"].to_numpy()
    values = clean["observations"]["value"].to_numpy(dtype=float)

    # Age per patient, for the MAR mechanism.
    ages: dict[str, float] = {}
    today = dt.date(2025, 6, 30)
    for row in clean["patients"].itertuples(index=False):
        born = dt.date.fromisoformat(row.birth_date)
        ages[row.patient_id] = (today - born).days / 365.25

    null_token_pool = ["", "N/A", "-", "NULL", "not recorded"]

    def _null_cell(idx: int) -> None:
        observations.at[idx, "value"] = str(rng.choice(null_token_pool))
        manifest.nulled.append(("observations", idx, "value"))

    # MCAR -- haemoglobin, flat 12%.
    hb_idx = np.flatnonzero(codes == vocab.Loinc.HEMOGLOBIN)
    for idx in hb_idx[rng.random(len(hb_idx)) < 0.12]:
        _null_cell(int(idx))
    manifest.missing_mechanism[vocab.Loinc.HEMOGLOBIN] = "MCAR"

    # MAR -- CA-125, ordered less often in younger women. Depends on age,
    # which IS recorded, so conditioning on age recovers the distribution.
    ca_idx = np.flatnonzero(codes == vocab.Loinc.CA125)
    for idx in ca_idx:
        pid = observations.at[int(idx), "patient_id"]
        p_missing = 0.55 if ages.get(pid, 40) < 35 else 0.08
        if rng.random() < p_missing:
            _null_cell(int(idx))
    manifest.missing_mechanism[vocab.Loinc.CA125] = "MAR (on age)"

    # MNAR -- BMI, missing more often when it is high. Depends on the value
    # that was deleted, so the observed mean is biased low and stays biased
    # after any conditional imputation.
    bmi_idx = np.flatnonzero(codes == vocab.Loinc.BMI)
    for idx in bmi_idx:
        p_missing = 0.45 if values[idx] > 28 else 0.06
        if rng.random() < p_missing:
            _null_cell(int(idx))
    manifest.missing_mechanism[vocab.Loinc.BMI] = "MNAR (on the BMI value itself)"

    # Light MCAR across the remaining vitals so the missingness map is not
    # suspiciously tidy.
    for code in (vocab.Loinc.LEUKOCYTES, vocab.Loinc.CREATININE, vocab.Loinc.TEMPERATURE):
        idxs = np.flatnonzero(codes == code)
        for idx in idxs[rng.random(len(idxs)) < 0.05]:
            _null_cell(int(idx))
        manifest.missing_mechanism[code] = "MCAR"

    return observations


def _inject_duplicates(patients: pd.DataFrame, encounters: pd.DataFrame,
                       rng: np.random.Generator, manifest: Manifest
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three kinds of repeated row, only two of which are duplicates.

      1. exact      the same registration submitted twice
      2. near       the same person re-registered at another site: new MRN,
                    name spelling variant, same date of birth
      3. legitimate the same person attending twice -- same MRN, different
                    encounter. NOT a duplicate, and a deduplicator that
                    removes it is destroying clinical history.

    Case 3 is the one that makes this a real test.
    """
    extra_patients, extra_encounters = [], []
    n = len(patients)

    # 1. exact duplicates
    for idx in rng.choice(patients.index, size=max(1, n // 40), replace=False):
        row = patients.loc[idx].to_dict()
        extra_patients.append(row)
        manifest.exact_duplicate_ids.append(str(row["patient_id"]))

    # 2. near duplicates under a fresh MRN
    for k, idx in enumerate(rng.choice(patients.index, size=max(1, n // 20), replace=False)):
        row = patients.loc[idx].to_dict()
        original_id = str(row["patient_id"])
        new_id = f"MRN9{80000 + k}"

        given = str(row["given_name"]).strip()
        family = str(row["family_name"]).strip()
        row = dict(row)
        row["patient_id"] = new_id
        row["given_name"] = _NAME_VARIANTS.get(given.title(), given)
        row["family_name"] = _NAME_VARIANTS.get(family.title(), family)
        row["site"] = str(rng.choice(_SITES))
        extra_patients.append(row)
        manifest.duplicate_of[new_id] = original_id

    # 3. legitimate repeat visits -- extra encounters, no extra patient row
    for k, idx in enumerate(rng.choice(patients.index, size=max(1, n // 15), replace=False)):
        pid = str(patients.at[idx, "patient_id"])
        base = encounters[encounters["patient_id"] == pid]
        if base.empty:
            continue
        row = base.iloc[0].to_dict()
        row["encounter_id"] = f"ENC9{70000 + k}"
        row["department"] = str(rng.choice(_DEPARTMENTS))
        extra_encounters.append(row)

    patients = pd.concat([patients, pd.DataFrame(extra_patients)], ignore_index=True)
    encounters = pd.concat([encounters, pd.DataFrame(extra_encounters)], ignore_index=True)

    # Shuffle so duplicates are not conveniently adjacent.
    patients = patients.sample(frac=1.0, random_state=int(rng.integers(0, 1 << 31))
                               ).reset_index(drop=True)
    return patients, encounters


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build(n_patients: int = 400, seed: int = 7,
          scan_ids: list[int] | None = None) -> Cohort:
    """Generate the cohort. Deterministic for a given seed.

    scan_ids are MMOTU OTU_2d image stems; the default covers the range the
    dataset ships so imaging_studies.scan_stem resolves against real files.
    """
    rng = np.random.default_rng(seed)
    if scan_ids is None:
        scan_ids = list(range(1, 1470))

    clean = _build_clean(rng, n_patients, scan_ids)
    manifest = Manifest()
    dirty = _corrupt(clean, rng, manifest)
    return Cohort(clean=clean, dirty=dirty, manifest=manifest)


if __name__ == "__main__":
    cohort = build()
    for name, df in cohort.dirty.items():
        print(f"{name:18s} {len(df):5d} rows  x {len(df.columns)} cols")
    print()
    for key, value in cohort.manifest.summary().items():
        print(f"{key:20s} {value}")
    print()
    for code, mech in cohort.manifest.missing_mechanism.items():
        print(f"  {vocab.LOINC[code]['display']:24s} {mech}")
