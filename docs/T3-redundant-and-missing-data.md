# T3 — Redundant data removal and missing data

> Syllabus: *"T3: Preprocessing of HER – Redundant data removal, Missing data"*

**Code:** [`backend/ehr/dedupe.py`](../backend/ehr/dedupe.py),
[`backend/ehr/missing.py`](../backend/ehr/missing.py) · **Page:** `/ehr#t3`

Two problems that both present as "bad rows" and need opposite treatments: duplicates
must be **merged**, absences must be **modelled**.

---

# Part 1 — Redundant data removal

## Three things that look like repetition; two are redundant

| | What it is | Treatment |
|---|---|---|
| **Exact duplicate** | same registration submitted twice | drop |
| **Near duplicate** | one person, two MRNs, two sites | **link** |
| **Repeat encounter** | one person attending twice | **keep — not redundant** |

The third case is why this is record linkage and not `drop_duplicates()`. A
deduplicator that collapses repeat visits is deleting clinical history, and it will
look like it is working. The generator injects all three so the pipeline is tested
against the case it must *not* act on.

## Why fuzzy matching is unavoidable

The near duplicates share no identifier. A patient re-registering at another site gets
a fresh MRN, and their name is re-typed — `Lakshmi`/`Laxmi`, `Priya`/`Prya`,
`Subramanian`/`Subramaniam`. Exact matching finds none of them.

## Blocking — what makes it tractable

All-pairs comparison is O(n²). For 420 patients that is 87,990 comparisons; for a real
cohort of 500,000 it is 125 **billion**.

Blocking compares only records agreeing on a cheap key — here, exact birth date:

```
all_pairs_would_be           87990
blocks                          22
comparisons_after_blocking      22
reduction                   100.0%
```

The cost is real and is counted, not hidden: a record whose **blocking key** is wrong is
never compared to anything, so the match is missed no matter how good the scorer is.
`unblocked_records` reports how many had no usable key. The key must therefore be a
field that is rarely mistyped and never legitimately changes — which is why date of
birth, not surname.

## Scoring

Weighted agreement across identifying fields:

```
0.45 × given-name similarity
0.45 × family-name similarity
0.10 × sex agreement
```

Names carry the weight because the block already fixes the birth date. Site
disagreement is deliberately *not* penalised — cross-site re-registration is the usual
cause of a duplicate, so disagreement there is weak evidence *for* a match.

Similarity is `difflib.SequenceMatcher` — a Ratcliff/Obershelp ratio, stdlib and
deterministic. `Laxmi`/`Lakshmi` scores 0.83, `Prya`/`Priya` 0.89. A production system
would use Jaro–Winkler, which weights a shared prefix more heavily and separates name
variants better.

## Transitive closure

`resolve` uses union-find. If A matches B and B matches C, all three are one person
even when A and C never scored above threshold. Skipping this leaves a half-merged
cluster, which is **worse** than not merging at all — the record is now split
unpredictably rather than predictably.

The surviving MRN is the lexicographically smallest in the cluster: arbitrary, but
*stable*, so a rerun produces the same survivor. A real system would keep the earliest
registration.

## Results, and the threshold

Against the generator manifest at threshold 0.80:

```
true_duplicates_injected   20
true_positives             20
false_positives             0
false_negatives             0
precision               1.000
recall                  1.000
```

The threshold was not chosen by eye. `threshold_sweep()` produces:

| threshold | tp | fp | fn | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.70 | 20 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| 0.80 | 20 | 0 | 0 | 1.000 | 1.000 | **1.000** |
| 0.82 | 19 | 0 | 1 | 1.000 | 0.950 | 0.974 |
| 0.88 | 17 | 0 | 3 | 1.000 | 0.850 | 0.919 |
| 0.95 | 15 | 0 | 5 | 1.000 | 0.750 | 0.857 |

0.80 sits at the top of the plateau rather than its middle, because the failure that
matters is a false merge.

### Held-out validation

Tuning and scoring on one cohort is overfitting, so it was checked on other seeds at
the same cohort size:

| seed | precision | recall | F1 |
|---|---|---|---|
| 7 (tuned on) | 1.000 | 1.000 | 1.000 |
| 99 (held out) | 1.000 | 0.950 | 0.974 |
| 123 (held out) | 1.000 | 0.950 | 0.974 |
| 2024 (held out) | 1.000 | 1.000 | 1.000 |
| 555 (held out) | 1.000 | **0.800** | 0.889 |

**Precision holds at 1.000 on every seed tried; recall ranges 0.80–1.00.** The spread is
the honest figure. The first two held-out seeds checked both landed at 0.950 and a
wider sweep was only run afterwards — quoting those two alone would have understated
the variance by a factor of three.

Every failure is a **missed match**, never a wrong merge, which is the direction that
matters ([see below](#why-both-error-types-matter-asymmetrically)). The misses are name
variants too distant for the Ratcliff/Obershelp ratio; Jaro–Winkler, which weights a
shared prefix more heavily, would recover most of them and is the obvious improvement.

Cohort size also matters: at `n_patients=300` only 15 duplicates are injected, so each
miss costs 6.7 points of recall instead of 5. Compare these figures only at equal `n`.

## Why both error types matter, asymmetrically

- A **missed** duplicate splits one person across two charts. An allergy recorded on
  one is invisible on the other.
- A **false** match merges two people. One patient inherits another's diagnosis.

The second is the more dangerous failure, which is why the threshold is placed where
precision stays at 1.000 even at some cost to recall.

---

# Part 2 — Missing data

## Two questions, in order

1. **Why** is it missing? The mechanism decides which repairs are valid.
2. **What** should replace it? Or nothing, if the mechanism forbids it.

Most pipelines skip the first and get the second wrong as a result.

## The three mechanisms (Rubin, 1976)

**MCAR** — Missing Completely At Random. Probability independent of everything. A
dropped tube. Complete-case analysis is unbiased, only wasteful.

**MAR** — Missing At Random. Probability depends on **observed** data. CA-125 is
ordered less often in younger women; missingness depends on age, and age is recorded.
Imputation conditioning on age recovers the distribution. *"At random" is a misleading
name* — it means random **given the observed data**.

**MNAR** — Missing Not At Random. Probability depends on the **unobserved value
itself**. BMI recorded less often when high. No imputation fixes this, because the
information needed is exactly what is absent.

**MAR and MNAR cannot be distinguished from the data.** The evidence that would
separate them is the missing values. It is a claim about how data was *collected*. This
cohort is synthetic, so the mechanism is known by construction — which is the only
reason the comparison below means anything.

## Injected mechanisms

| Observation | Mechanism | Rule |
|---|---|---|
| Haemoglobin | MCAR | flat 12% |
| CA-125 | MAR on age | 55% if age < 35, else 8% |
| BMI | MNAR on BMI | 45% if BMI > 28, else 6% |

## Diagnostics — three distinct signatures

`little_style_correlation` correlates the missingness indicator against every other
observed value. It can **refute** MCAR and separate it from MAR. It cannot separate MAR
from MNAR.

| Target | Strongest correlate | r | Reading |
|---|---|---|---|
| CA-125 | age | **−0.426** | MAR detected — and the sign is right, missing more at younger ages |
| BMI | body weight | **0.421** | MNAR leaking through a proxy |
| Haemoglobin | age | 0.071 | noise — consistent with MCAR |

The BMI row is the subtle one. BMI's missingness depends on BMI, which is unobserved —
but weight correlates with BMI, so the mechanism is partly visible through it. That is
how MNAR usually announces itself: not directly, but through a proxy.

> **Implementation note.** The covariate join was initially done on `patient_id` and
> silently lost age for 11 encounters whose patient row had been merged away by the
> deduplicator — it surfaced as 2.8% unexplained missingness in a column that should
> have had none. After dedupe, child rows must join on `master_patient_id`. Fixed in
> `attach_covariates`; recorded here because it is exactly the kind of bug that would
> otherwise quietly bias the diagnostic.

## Why not just drop incomplete rows

```
encounters             400
complete_rows          171
complete_case_loss   57.2%
```

Listwise deletion discards **57.2%** of the cohort. Under MCAR that is merely wasteful.
Under MAR or MNAR it is also **biased**, because the survivors are not a random sample.

## Imputation, scored against the deleted values

Four strategies, scored on the cells that were actually removed. RMSE and **bias**
together, because they fail differently — a method can post a respectable RMSE and
still be systematically low, which is what MNAR produces and what a single error number
hides.

### Haemoglobin — MCAR

| strategy | RMSE | bias |
|---|---|---|
| mean | 1.419 | −0.118 |
| median | 1.428 | −0.198 |
| knn | 1.635 | −0.162 |
| iterative | 1.417 | −0.127 |

All near-unbiased, as theory predicts. Nothing to gain from a sophisticated method.

### CA-125 — MAR

| strategy | RMSE | bias |
|---|---|---|
| mean | 25.675 | +0.377 |
| **median** | 27.153 | **−8.844** |
| knn | 30.752 | +2.994 |
| iterative | 25.671 | +0.372 |

The median is badly biased here despite being the usual "robust" default — CA-125 is
log-normal, so its median sits far below its mean and imputing the median drags the
whole distribution down. A reminder that robustness is relative to a distribution.

### BMI — MNAR

| strategy | RMSE | bias |
|---|---|---|
| mean | 7.501 | −6.200 |
| median | 7.686 | −6.423 |
| knn | 3.798 | −2.616 |
| **iterative** | **1.535** | **−0.365** |

True mean 30.2; mean-imputation yields 24.0. Every simple strategy is biased low —
exactly the MNAR signature, because the values that went missing were the large ones.

### The MNAR result is not what it appears to be

Iterative (MICE-style) imputation nearly eliminates the bias. **This is not MICE
solving MNAR.**

BMI is *derivable* from height and weight, both of which are present. The information
was never actually missing; MICE finds the deterministic relationship
`BMI = weight/(height/100)²` and reconstructs it. Any method with access to both
columns would.

For a genuinely unobservable MNAR quantity — a symptom severity that goes unrecorded
*because* it was severe — no imputation recovers the distribution. The honest options
are to model the mechanism explicitly (selection models, pattern-mixture models) or to
declare the limitation. Reporting the BMI result as evidence that MICE handles MNAR
would be the wrong lesson to take from this table, and it is the one the numbers
superficially support.

---

## What this does not do

- **No multiple imputation.** `IterativeImputer` runs with `sample_posterior=False`, so
  it produces a single point estimate. Proper MI generates *m* datasets and pools with
  Rubin's rules, which is what propagates imputation uncertainty into the final
  standard errors. Single imputation understates them.
- **No MNAR-specific models.** No selection or pattern-mixture model is fitted.
- **Fuzzy matching is Ratcliff/Obershelp**, not Jaro–Winkler or a trained
  Fellegi–Sunter model.

---

Previous: [T2](T2-standardization-and-cleaning.md) · Next: [T4](T4-image-processing.md)
