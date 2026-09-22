# T1 — Understanding EHR

> Syllabus: *"T1: Understanding HER"* — Unit-1, *Electronic Health Records, Components
> of HER, Coding Systems, Benefits of HER, Barriers to Adopting HER, Challenges of
> Using EHR Data.*

**Code:** [`backend/ehr/vocab.py`](../backend/ehr/vocab.py),
[`backend/ehr/generate.py`](../backend/ehr/generate.py) · **Page:** `/ehr#t1`

---

## 1. What an EHR actually is

Not "a database of patients". An EHR is a **longitudinal record of care events**, and
its structure follows from that. Four things have to be representable:

| Component | Question it answers | Table here |
|---|---|---|
| Demographics | Who is the patient? | `patients` |
| Encounters | When were they seen, and by whom? | `encounters` |
| Observations | What was measured? | `observations` |
| Diagnoses / imaging | What was found? | `imaging_studies` |

The real ones add orders, medications, allergies, notes and billing. This cohort models
the four that the rest of the project needs, which is a scope decision rather than a
claim that the others do not matter.

### Why `observations` is long, not wide

One row per measurement — `(encounter_id, loinc, value, unit)` — rather than one column
per test. This is how every real clinical data warehouse stores observations, for a
reason that is easy to miss: the set of tests is **open**. A wide table needs a schema
migration every time a new assay is introduced, and it is mostly nulls, because no
patient gets every test.

The cost is that cross-field rules ("systolic must exceed diastolic") need a pivot
first. [`clean.py`](../backend/ehr/clean.py) does exactly that in `_pivot_by_encounter`,
and the comment there says why the long form is right for storage and wrong for that
one job.

## 2. Coding systems — the part that makes it interoperable

A record is only exchangeable if its contents mean the same thing to the receiving
system. That is what code systems are for, and this project uses three.

### ICD-10 — diagnoses

Classification of diseases. Used here for what was found on the scan:

| Code | Meaning | MMOTU classes mapped to it |
|---|---|---|
| `N83.20` | Unspecified ovarian cyst | simple cyst |
| `N80.1` | Endometriosis of ovary | chocolate cyst |
| `D27.9` | Benign neoplasm of ovary | serous/mucinous cystadenoma, teratoma, theca cell |
| `C56.9` | Malignant neoplasm of ovary | high grade serous |
| `Z12.73` | Screening encounter, ovarian | normal ovary |

The mapping source is the MMOTU dataset's own eight tumour-type labels. That is the
point: those strings are a **local vocabulary**. They mean something to the dataset
authors and nothing to any other system, so mapping them to ICD-10 is a real
terminology-binding exercise rather than a made-up one. It is also the exact task T2
performs — see `DIAGNOSIS_SYNONYMS` in `vocab.py`, which carries 26 clinician spellings
that all have to collapse onto 5 codes.

Note that the segmentation model collapses everything to a single `lesion` class while
the chart keeps the finer label. That asymmetry is deliberate: a record that says only
"lesion" is useless to a clinician, and a one-class model is the right choice for a
small dataset.

### LOINC — observations

Identifies *what was measured*. Eleven concepts here, each with a canonical unit and a
physiological range:

`8302-2` height · `29463-7` weight · `39156-5` BMI · `8480-6` systolic ·
`8462-4` diastolic · `8867-4` heart rate · `8310-5` temperature ·
`718-7` haemoglobin · `6690-2` leukocytes · `2160-0` creatinine · `10334-1` CA-125

CA-125 is the clinically interesting one — a serum marker elevated in ovarian
malignancy, which is why the generator draws it log-normally with a higher mean for the
high-grade-serous class. It is also the field the MAR missingness mechanism acts on,
because it is ordered selectively rather than routinely.

The `canonical_unit` and `plausible` fields are not decoration. `canonical_unit` is the
target of every conversion in T2; `plausible` is the range T2's cleaning uses to
separate impossible values from merely extreme ones. Keeping them beside the code means
the rule and the concept cannot drift apart.

### SNOMED CT — findings

**The SNOMED entries in `vocab.py` are labelled placeholders, not real concept ids.**
They demonstrate the three-system structure and are confined to `SNOMED_FINDINGS` so
nothing else depends on them. Replace them from the official browser before any use
beyond this coursework. Fabricating terminology codes that *look* authoritative is
worse than omitting them — a wrong code is wrong and looks complete.

## 3. Benefits, barriers, challenges — as the code encounters them

The syllabus lists these as prose topics. They are more convincing as things the
pipeline actually runs into.

**Benefits** the structure enables: longitudinal view across encounters; decision
support, which is only possible because observations are coded rather than free text;
population queries; and the one this project depends on — linking imaging to clinical
context, so a lesion measurement sits next to the CA-125 that was drawn the same week.

**Barriers** visible in the data: three sites use three different unit conventions
(`_SITE_UNITS`), three different local test names per concept (`_LOCAL_TEST_NAMES`),
and four different date formats. None is a mistake; each is a locally reasonable
convention. Interoperability is expensive precisely because nobody is doing anything
wrong.

**Challenges** of using the data, each with a concrete instance downstream:

| Challenge | Where it shows up |
|---|---|
| Missingness is not random | CA-125 is 24% absent, concentrated in younger patients — [T3](T3-redundant-and-missing-data.md) |
| Identity is not stable | One person, two MRNs, at two sites — [T3](T3-redundant-and-missing-data.md) |
| Values arrive as text | `"12,5"`, `"45 kg"`, `"N/A"` in a numeric column — [T2](T2-standardization-and-cleaning.md) |
| Impossible values pass validation | Height 1580 cm; a missing decimal point — [T2](T2-standardization-and-cleaning.md) |
| Units are implicit | 58.65 is a plausible height in inches and impossible in cm — [T2](T2-standardization-and-cleaning.md) |

## 4. The cohort

`generate.build(n_patients=400, seed=7)` produces, deterministically:

```
patients          430 rows   (400 people + 10 exact dupes + 20 re-registrations)
encounters        426 rows   (400 + 26 legitimate repeat visits)
observations     4400 rows   (11 per encounter)
imaging_studies   400 rows   (one MMOTU scan each)
```

Then it corrupts a copy and records what it did: 303 cells nulled, 38 values made
physiologically impossible, 20 forged near-duplicates, 10 exact duplicates.

Everything is fabricated. The names come from a deliberately small pool so that
collisions occur — which gives the fuzzy matcher in T3 genuine false-positive pressure
instead of an artificially easy problem.

### Why not Synthea or a real de-identified extract

Both were considered. Synthea produces richer records with native ICD-10/LOINC/SNOMED
coding, and a public clinical CSV would be authentically messy. Neither can do the one
thing this design needs: **hand back the uncorrupted original**. Without it, "KNN
imputation beat the median by 0.8 BMI points" is unverifiable, and "the linkage found
19 of 20 duplicates" cannot be said at all. Controlling the defects is what turns T2
and T3 from a description into a measurement.

The trade-off is honest and worth stating: a synthetic cohort cannot surprise you. Real
data contains defect *categories* nobody thought to inject, and a pipeline tuned only
on generated data will meet them unprepared.

---

Next: [T2 — standardization and cleaning](T2-standardization-and-cleaning.md)
