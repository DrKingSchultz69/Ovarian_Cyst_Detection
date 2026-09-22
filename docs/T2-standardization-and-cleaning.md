# T2 — Standardization and data cleaning

> Syllabus: *"T2: Preprocessing of HER – Standardization, Data Cleaning"*

**Code:** [`backend/ehr/standardize.py`](../backend/ehr/standardize.py),
[`backend/ehr/clean.py`](../backend/ehr/clean.py) · **Page:** `/ehr#t2`

---

## The distinction that organises everything

These are two jobs, and conflating them is the commonest mistake in the topic.

| | Standardization | Cleaning |
|---|---|---|
| Changes | representation | which values are trusted |
| Preserves | meaning, always | — it decides some values are wrong |
| Example | `58.65 in` → `148.97 cm` | `1580 cm` → quarantined |
| Reversible | yes, raw kept in `*_raw` | recorded in a flags table |

**Order is not negotiable.** A plausibility check applied before unit conversion
rejects every height recorded at the imperial site — 58.65 looks impossible next to a
40–230 cm rule, and it is a perfectly normal 4′11″ patient. Run standardization first,
always.

---

## Part 1 — Standardization

Four kinds, all in `standardize.py`.

### Syntactic — dates

Four formats in the source, one on the way out:

```
2007-06-29   %Y-%m-%d      site RGH
29/06/2007   %d/%m/%Y      site CMC
06-29-2007   %m-%d-%Y      site APX
29 Jun 2007  %d %b %Y      long form
```

`parse_date` tries each in order and returns ISO-8601.

**The ambiguity is real and is documented rather than hidden.** `03/04/1990` is 3 April
under `%d/%m` and 4 March under `%m/%d`, and *nothing in the string resolves it*. In
this cohort the separator happens to disambiguate, because the generator gives each
site a distinct layout — but that is a property of this data, not a general rule. Real
intake needs the source system documented. The `dayfirst_default` parameter exists to
make the assumption explicit instead of silent.

### Unit — the substantive part

18 distinct units in, 10 canonical out, zero unconvertible.

`UNIT_ALIASES` maps written forms (`"lbs"`, `"pounds"`, `"LB"`) onto UCUM-flavoured
spellings; `UNIT_FACTORS` holds multiplicative conversions onto the canonical unit.

Two details worth the attention:

**Temperature is deliberately absent from the factor table.** °F → °C is
`(x − 32) × 5/9` — affine, not multiplicative. A lookup table of multipliers cannot
express it, and forcing it in would produce a silently wrong number for every
temperature at the imperial site. It is special-cased in `convert_value` instead.

**A unit glued to the value overrides the unit column.** `"45 kg"` in a numeric field
whose unit column says `lbs` is a conflict, and `parse_numeric` returns the embedded
unit so it wins. The reasoning: both were typed by the same person at the same moment,
whereas the unit column is often a template default nobody looked at. 78 values in this
cohort take that path.

`convert_value` returns `None` rather than passing the value through when no conversion
is known. A pass-through would mean silently emitting a pound value into a kilogram
column — which is the failure mode that this whole exercise exists to prevent.

### Semantic — local names to codes

33 local test names collapse onto 11 LOINC concepts. `Hb`, `HGB` and `Haemoglobin` are
one thing; so are `SBP`, `Systolic BP` and `BP systolic`. Lookup goes through
`normalise_key` (lower-case, collapse whitespace, strip punctuation) so the dictionaries
carry only genuine spelling variants rather than every capitalisation.

Similarly 26 free-text diagnoses → 5 ICD-10 codes.

**Unmapped values are kept with a null code, never dropped or guessed.** An incorrectly
coded diagnosis is worse than an uncoded one: it is wrong *and* it looks complete. The
report counts them by name so a reviewer can extend the mapping.

### Categorical — value sets

8 spellings of sex (`F`, `f`, `FEMALE`, `W`, `woman`, `2`, …) → 1 FHIR
administrative-gender value.

Names are trimmed, whitespace-collapsed and title-cased. **Title-casing is lossy** —
`McDonald` and `van der Berg` come out wrong — and it is applied only because the
matcher in T3 needs a stable form, and it compares case-folded anyway. The original
survives in `given_name_raw` / `family_name_raw`. An EHR transform that destroys the
source value cannot be audited, which is the main argument for doing this in a pipeline
rather than by hand.

### Measured result

```
rows                       4400
distinct_local_test_names    33  →  11 LOINC concepts
distinct_units_before        18  →  10 canonical
values_with_embedded_unit    78
unmapped_test_names        none
unconvertible_unit_pairs   none
sex_distinct_before           8  →   1
distinct_diagnosis_text      26  →   5 ICD-10 codes
birth_dates_unparsed          0
```

---

## Part 2 — Cleaning

Three kinds of wrong, which must not be treated alike.

| Kind | Example | Action |
|---|---|---|
| **Impossible** | height 1580 cm | quarantine |
| **Inconsistent** | systolic 70 with diastolic 120 | quarantine both |
| **Extreme** | CA-125 2400 U/mL | **flag only** |

### Rule 1 — physiological plausibility

Per-LOINC ranges from `vocab.LOINC`. Runs only after unit conversion.

Detected all 30 injected impossible values:

| Injected | Meaning |
|---|---|
| height 1580 | missing decimal point |
| weight 0.0 | scale never read |
| heart rate 999 | sentinel typed into a numeric field |
| creatinine −1.2 | sign slip |
| temperature 3.68 | decimal shifted |

### Rule 2 — cross-field consistency

Individually plausible, jointly contradictory. Caught all 8 injected systolic/diastolic
transpositions, quarantining 16 cells.

**Both values are quarantined, not swapped back.** Swapping assumes the pair was
transposed rather than one value mistyped, and that assumption is not recoverable from
the data. Quarantining is the honest response to "we know this is wrong and not which
part".

BMI gets a different treatment: where recorded BMI disagrees with
`weight / (height/100)²` by more than 1.5 but all three are individually plausible, it
is **flagged, not repaired** — any of the three could be the wrong one. Only the
derivable case is filled: missing BMI with good height and weight.

(This check only works *after* unit conversion, which is a neat demonstration of the
ordering rule — BMI is always reported in kg/m² while height and weight arrive in
inches and pounds at one site.)

### Rule 3 — statistical outliers, flagged only

Robust z-score on median and MAD rather than mean and standard deviation:

```
robust_z = 0.6745 × (x − median) / MAD
```

Mean and SD are the wrong estimators here because a handful of extreme values inflates
the SD enough to hide themselves. `0.6745` is the 75th-percentile z of a normal, making
MAD a consistent estimator of σ under normality.

**Nothing is deleted on statistical grounds.** In an ovarian-pathology cohort the
extreme CA-125 values are the positive cases. An IQR filter here removes exactly the
patients the study is about — this is the single most important judgement in the
module, and it is why `action` distinguishes `quarantined` from `flagged`.

### Measured result

```
flags_raised          77
values_quarantined    46      (30 plausibility + 16 BP consistency)
values_flagged_only   31      (outliers — all retained)
```

### Every rejection is recorded

`clean_all` returns a flags table: rule, table, row, column, value, action, reason. A
pipeline that silently drops rows is unauditable, and in a clinical context that is
disqualifying. The `/ehr` page renders this table in full.

---

## What this does not do

- **No fuzzy matching of unmapped test names.** A name not in the dictionary stays
  unmapped. Auto-mapping `"Hb (capillary)"` to venous haemoglobin by string similarity
  would be a silent clinical error.
- **No imputation.** Quarantined values become `NaN` and are handled by
  [T3](T3-redundant-and-missing-data.md). That is deliberate: to every downstream model
  an impossible value and an absent one are the same thing, and pretending otherwise
  invents information.
- **Dates are not disambiguated beyond format order.** See above.

---

Previous: [T1](T1-understanding-ehr.md) · Next: [T3](T3-redundant-and-missing-data.md)
