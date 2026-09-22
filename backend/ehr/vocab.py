"""Controlled vocabularies -- the coding-systems half of T1.

An EHR is only interoperable because the things inside it are coded. This
module holds the small subsets of ICD-10, LOINC and SNOMED CT that the OvaScan
cohort actually uses, plus the messy local spellings each one has to absorb.

Why this is separate from standardize.py: the *mappings* are reference data and
the *act of mapping* is a transform. Keeping them apart means a reviewer can
audit the terminology without reading any pandas.

  ICD-10  diagnoses          -- what is wrong with the patient
  LOINC   observations       -- what was measured
  SNOMED  clinical findings  -- what was observed

ACCURACY NOTE
    The ICD-10 and LOINC codes below are real and verifiable. The SNOMED CT
    entries are ILLUSTRATIVE PLACEHOLDERS -- they show the three-system
    structure but are not real concept ids, and must be replaced from the
    official SNOMED browser before any use outside this coursework. They are
    confined to SNOMED_FINDINGS so nothing else depends on them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ICD-10 -- diagnoses
#
# MMOTU labels its images with eight tumour types. Those strings are a *local*
# vocabulary: they mean something to the dataset authors and nothing to any
# other system. Mapping them onto ICD-10 is the canonical T2 exercise, so the
# dataset taxonomy is kept here verbatim as the mapping source.
# ---------------------------------------------------------------------------

ICD10: dict[str, str] = {
    "N83.20": "Unspecified ovarian cyst",
    "N83.0": "Follicular cyst of ovary",
    "N83.1": "Corpus luteum cyst",
    "N80.1": "Endometriosis of ovary",
    "D27.9": "Benign neoplasm of unspecified ovary",
    "C56.9": "Malignant neoplasm of unspecified ovary",
    "Z12.73": "Encounter for screening for malignant neoplasm of ovary",
}

# MMOTU class id -> (local label, ICD-10). The segmentation model collapses
# everything to one `lesion` class; the clinical record keeps the finer label,
# because a chart that says only "lesion" is useless to a clinician.
MMOTU_CLASS_TO_ICD10: dict[int, tuple[str, str]] = {
    0: ("chocolate cyst", "N80.1"),
    1: ("serous cystadenoma", "D27.9"),
    2: ("teratoma", "D27.9"),
    3: ("theca cell tumor", "D27.9"),
    4: ("simple cyst", "N83.20"),
    5: ("normal ovary", "Z12.73"),
    6: ("mucinous cystadenoma", "D27.9"),
    7: ("high grade serous", "C56.9"),
}

# The same diagnosis as typed by eight different people at three sites.
# Every key here must normalise to the same ICD-10 code.
DIAGNOSIS_SYNONYMS: dict[str, str] = {
    "chocolate cyst": "N80.1",
    "chocolate-cyst": "N80.1",
    "endometrioma": "N80.1",
    "endometriotic cyst": "N80.1",
    "serous cystadenoma": "D27.9",
    "serous cyst adenoma": "D27.9",
    "teratoma": "D27.9",
    "dermoid": "D27.9",
    "dermoid cyst": "D27.9",
    "mature teratoma": "D27.9",
    "theca cell tumor": "D27.9",
    "theca cell tumour": "D27.9",
    "thecoma": "D27.9",
    "mucinous cystadenoma": "D27.9",
    "mucinous cyst adenoma": "D27.9",
    "simple cyst": "N83.20",
    "simple ovarian cyst": "N83.20",
    "ovarian cyst": "N83.20",
    "cyst": "N83.20",
    "follicular cyst": "N83.0",
    "corpus luteum cyst": "N83.1",
    "high grade serous": "C56.9",
    "hgsc": "C56.9",
    "high grade serous carcinoma": "C56.9",
    "normal ovary": "Z12.73",
    "normal": "Z12.73",
    "no abnormality detected": "Z12.73",
    "nad": "Z12.73",
}

# ---------------------------------------------------------------------------
# LOINC -- observations
#
# canonical_unit is the unit every value is converted TO. A value arriving in
# another unit is a standardisation problem, not a data-entry error: it must be
# converted, never discarded. `plausible` is the physiological range used by
# clean.py to separate real outliers from impossible values.
# ---------------------------------------------------------------------------


class Loinc:
    HEIGHT = "8302-2"
    WEIGHT = "29463-7"
    BMI = "39156-5"
    SYSTOLIC = "8480-6"
    DIASTOLIC = "8462-4"
    HEART_RATE = "8867-4"
    TEMPERATURE = "8310-5"
    HEMOGLOBIN = "718-7"
    LEUKOCYTES = "6690-2"
    CREATININE = "2160-0"
    CA125 = "10334-1"


LOINC: dict[str, dict[str, object]] = {
    Loinc.HEIGHT: {"display": "Body height", "canonical_unit": "cm", "plausible": (40.0, 230.0)},
    Loinc.WEIGHT: {"display": "Body weight", "canonical_unit": "kg", "plausible": (2.0, 350.0)},
    Loinc.BMI: {"display": "Body mass index", "canonical_unit": "kg/m2", "plausible": (8.0, 90.0)},
    Loinc.SYSTOLIC: {"display": "Systolic blood pressure", "canonical_unit": "mm[Hg]", "plausible": (50.0, 260.0)},
    Loinc.DIASTOLIC: {"display": "Diastolic blood pressure", "canonical_unit": "mm[Hg]", "plausible": (25.0, 160.0)},
    Loinc.HEART_RATE: {"display": "Heart rate", "canonical_unit": "/min", "plausible": (25.0, 220.0)},
    Loinc.TEMPERATURE: {"display": "Body temperature", "canonical_unit": "Cel", "plausible": (30.0, 43.0)},
    Loinc.HEMOGLOBIN: {"display": "Hemoglobin", "canonical_unit": "g/dL", "plausible": (3.0, 22.0)},
    Loinc.LEUKOCYTES: {"display": "Leukocytes", "canonical_unit": "10*9/L", "plausible": (0.5, 100.0)},
    Loinc.CREATININE: {"display": "Creatinine", "canonical_unit": "mg/dL", "plausible": (0.1, 15.0)},
    Loinc.CA125: {"display": "Cancer Ag 125", "canonical_unit": "U/mL", "plausible": (0.0, 10000.0)},
}

# Local test names as they appear on requisition forms. Keys are already
# normalised (lower case, single spaces), so only genuine spelling variants
# need an entry -- not every capitalisation.
LOCAL_TEST_TO_LOINC: dict[str, str] = {
    "height": Loinc.HEIGHT, "ht": Loinc.HEIGHT, "body height": Loinc.HEIGHT,
    "stature": Loinc.HEIGHT,
    "weight": Loinc.WEIGHT, "wt": Loinc.WEIGHT, "body weight": Loinc.WEIGHT,
    "mass": Loinc.WEIGHT,
    "bmi": Loinc.BMI, "body mass index": Loinc.BMI, "quetelet index": Loinc.BMI,
    "sbp": Loinc.SYSTOLIC, "systolic": Loinc.SYSTOLIC, "systolic bp": Loinc.SYSTOLIC,
    "bp systolic": Loinc.SYSTOLIC,
    "dbp": Loinc.DIASTOLIC, "diastolic": Loinc.DIASTOLIC,
    "diastolic bp": Loinc.DIASTOLIC, "bp diastolic": Loinc.DIASTOLIC,
    "hr": Loinc.HEART_RATE, "pulse": Loinc.HEART_RATE, "heart rate": Loinc.HEART_RATE,
    "temp": Loinc.TEMPERATURE, "temperature": Loinc.TEMPERATURE,
    "body temperature": Loinc.TEMPERATURE,
    "hb": Loinc.HEMOGLOBIN, "hgb": Loinc.HEMOGLOBIN, "haemoglobin": Loinc.HEMOGLOBIN,
    "hemoglobin": Loinc.HEMOGLOBIN,
    "wbc": Loinc.LEUKOCYTES, "leukocytes": Loinc.LEUKOCYTES,
    "white cell count": Loinc.LEUKOCYTES, "tlc": Loinc.LEUKOCYTES,
    "creat": Loinc.CREATININE, "creatinine": Loinc.CREATININE,
    "s creatinine": Loinc.CREATININE,
    "ca125": Loinc.CA125, "ca 125": Loinc.CA125, "ca-125": Loinc.CA125,
    "cancer antigen 125": Loinc.CA125, "ca125 serum": Loinc.CA125,
}

# ---------------------------------------------------------------------------
# Units
#
# UNIT_FACTORS holds multiplicative conversions onto the canonical unit.
# Temperature is deliberately absent: F->C is affine, not multiplicative, and
# is special-cased in standardize.py.
# ---------------------------------------------------------------------------

UNIT_ALIASES: dict[str, str] = {
    "cms": "cm", "centimeter": "cm", "centimetre": "cm", "centimeters": "cm",
    "m": "m", "meter": "m", "metre": "m", "meters": "m",
    "in": "[in_i]", "inch": "[in_i]", "inches": "[in_i]",
    "kgs": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg",
    "lb": "[lb_av]", "lbs": "[lb_av]", "pound": "[lb_av]", "pounds": "[lb_av]",
    "g": "g", "gram": "g", "grams": "g",
    "mmhg": "mm[Hg]", "mm hg": "mm[Hg]",
    "bpm": "/min", "beats/min": "/min", "min-1": "/min",
    "c": "Cel", "celsius": "Cel", "degc": "Cel", "deg c": "Cel",
    "f": "[degF]", "fahrenheit": "[degF]", "degf": "[degF]", "deg f": "[degF]",
    "g/l": "g/L", "gm/dl": "g/dL", "gms/dl": "g/dL", "g/dl": "g/dL",
    "10^9/l": "10*9/L", "10e9/l": "10*9/L", "x10^9/l": "10*9/L",
    "/ul": "/uL", "cells/ul": "/uL", "per ul": "/uL",
    "umol/l": "umol/L", "mg/dl": "mg/dL",
    "u/ml": "U/mL", "iu/ml": "U/mL", "ku/l": "kU/L", "u/l": "U/L",
}

UNIT_FACTORS: dict[tuple[str, str], float] = {
    ("m", "cm"): 100.0,
    ("[in_i]", "cm"): 2.54,
    ("[lb_av]", "kg"): 0.45359237,
    ("g", "kg"): 0.001,
    ("g/L", "g/dL"): 0.1,
    ("/uL", "10*9/L"): 0.001,        # 1000 /uL == 1 x10^9/L
    ("umol/L", "mg/dL"): 0.0113122,  # creatinine-specific molar mass
    ("kU/L", "U/mL"): 1.0,           # dimensionally identical for CA-125
    ("U/L", "U/mL"): 0.001,
}

# ---------------------------------------------------------------------------
# Categorical value sets
# ---------------------------------------------------------------------------

# Administrative gender, after the FHIR value set. The cohort is
# gynaecological, so the clinically meaningful values are female and
# other/unknown -- but the raw feed still carries every spelling a
# registration clerk can produce.
SEX_SYNONYMS: dict[str, str] = {
    "f": "female", "fem": "female", "female": "female", "w": "female",
    "woman": "female", "girl": "female", "2": "female",
    "m": "male", "male": "male", "man": "male", "boy": "male", "1": "male",
    "o": "other", "other": "other", "x": "other",
    "u": "unknown", "unk": "unknown", "unknown": "unknown", "0": "unknown",
    "": "unknown", "n/a": "unknown", "na": "unknown", "-": "unknown",
    "null": "unknown", "none": "unknown", "?": "unknown", "nil": "unknown",
}

# Tokens that mean "no value recorded". Every one of these must become NaN
# rather than the literal string, or the column silently stays dtype=object
# and every numeric operation downstream fails.
NULL_TOKENS: frozenset[str] = frozenset({
    "", "-", "--", "n/a", "na", "nan", "null", "none", "nil", "?", "unknown",
    "not recorded", "not done", "pending", "#n/a", ".",
})

# SNOMED CT -- PLACEHOLDERS, see the accuracy note at the top of this file.
SNOMED_FINDINGS: dict[str, str] = {
    "pelvic pain": "ILLUSTRATIVE-pelvic-pain",
    "abdominal distension": "ILLUSTRATIVE-abdominal-distension",
    "menstrual irregularity": "ILLUSTRATIVE-menstrual-irregularity",
    "asymptomatic": "ILLUSTRATIVE-asymptomatic",
}

# ---------------------------------------------------------------------------


def normalise_key(text: object) -> str:
    """Lower-case, collapse whitespace, strip surrounding punctuation.

    Every lookup in this module goes through here, so the dictionaries above
    carry only genuine spelling variants rather than every combination of
    capitalisation and stray spacing.
    """
    if text is None:
        return ""
    s = " ".join(str(text).strip().lower().split())
    return s.strip(".,:;_")


def canonical_unit(loinc_code: str) -> str:
    """The unit every value for this observation is converted to."""
    return str(LOINC[loinc_code]["canonical_unit"])


def plausible_range(loinc_code: str) -> tuple[float, float]:
    """Physiological bounds. Outside these a value is impossible, not extreme."""
    lo, hi = LOINC[loinc_code]["plausible"]  # type: ignore[misc]
    return float(lo), float(hi)
