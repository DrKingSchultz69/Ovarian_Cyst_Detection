"""T3 part two -- missing data.

Two questions, in order. Most pipelines skip the first and get the second
wrong as a result.

  1. WHY is it missing?  The mechanism decides which repairs are valid.
  2. WHAT should replace it?  Or nothing, if the mechanism forbids it.

MECHANISMS (Rubin, 1976)

  MCAR  Missing Completely At Random. The probability of being missing is
        independent of everything, observed or not. A dropped tube, a lost
        form. Complete-case analysis is unbiased -- just wasteful.

  MAR   Missing At Random. Probability depends on OBSERVED data. CA-125 is
        ordered less often in younger women, so missingness depends on age,
        and age is recorded. Imputation that conditions on age recovers the
        distribution. "At random" is a misleading name: it means random
        *given the observed data*, not random.

  MNAR  Missing Not At Random. Probability depends on the UNOBSERVED value
        itself. BMI recorded less often when it is high. No imputation fixes
        this, because the information needed is exactly what is absent. It can
        only be modelled with an explicit assumption, or declared as a
        limitation.

Distinguishing MAR from MNAR is NOT possible from the data alone -- the
evidence that would separate them is the missing values. It is a claim about
how the data was collected. This cohort is synthetic, so the mechanism is
known by construction, and that is what makes the comparison below meaningful.

WHAT THIS MODULE MEASURES
    Because generate.py recorded every deleted value, each imputation strategy
    can be scored against the truth it is trying to reconstruct. The headline
    result is that the ranking of strategies differs by mechanism, and that
    every strategy fails on MNAR in the same direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import vocab
from .standardize import Report

# sklearn's iterative imputer is still behind an explicit opt-in import.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def to_wide(observations: pd.DataFrame) -> pd.DataFrame:
    """Long observations -> one row per encounter, one column per LOINC.

    Imputation needs the wide form: every method here estimates a value from
    the other measurements taken at the same encounter.
    """
    numeric = observations.copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    wide = numeric.pivot_table(
        index="encounter_id", columns="loinc", values="value", aggfunc="first"
    )
    wide.columns = [str(c) for c in wide.columns]
    return wide


def attach_covariates(wide: pd.DataFrame, encounters: pd.DataFrame,
                      patients: pd.DataFrame,
                      columns: tuple[str, ...] = ("age_years",)) -> pd.DataFrame:
    """Join patient-level columns onto the per-encounter matrix.

    Necessary, not cosmetic: a MAR mechanism can depend on any recorded
    variable, and the ones that drive it are often demographic rather than
    clinical. CA-125 here is MAR *on age*, and age lives in the patients
    table -- so a diagnostic run on the observation matrix alone cannot see
    the mechanism at all and will wrongly suggest MCAR.
    """
    # Join on the SURVIVING identifier where dedupe has run. Joining on the
    # raw patient_id silently drops the covariate for every encounter whose
    # patient row was merged away -- the rows are still there, the demographics
    # are not, and it shows up as unexplained missingness in the covariate.
    key = "master_patient_id" if (
        "master_patient_id" in encounters.columns
        and "master_patient_id" in patients.columns
    ) else "patient_id"

    link = encounters[["encounter_id", key]].drop_duplicates("encounter_id")
    keep = [key] + [c for c in columns if c in patients.columns]
    merged = (link.merge(patients[keep].drop_duplicates(key), on=key, how="left")
                  .set_index("encounter_id"))

    out = wide.join(merged.drop(columns=key), how="left")
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def profile(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness, ordered worst first."""
    rows = []
    for col in wide.columns:
        n_missing = int(wide[col].isna().sum())
        rows.append({
            "loinc": col,
            "display": vocab.LOINC.get(col, {}).get("display", col),
            "n": len(wide),
            "missing": n_missing,
            "missing_pct": round(100.0 * n_missing / max(len(wide), 1), 1),
            "mean_observed": round(float(wide[col].mean()), 2) if n_missing < len(wide) else None,
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False, ignore_index=True)


def little_style_correlation(wide: pd.DataFrame, target: str) -> pd.DataFrame:
    """Evidence for or against MCAR, without claiming to prove it.

    If whether `target` is missing correlates with the VALUE of other observed
    columns, the data is not MCAR. This is the diagnostic that is actually
    available: it can refute MCAR, and it can distinguish MAR from MCAR. It
    cannot separate MAR from MNAR, because that distinction depends on the
    values that were never recorded.

    (Little's MCAR test proper is a chi-square over the missingness patterns;
    this is the per-column correlation that gives the same signal in a form
    that is readable without a test statistic.)
    """
    indicator = wide[target].isna().astype(float)
    rows = []
    for col in wide.columns:
        if col == target:
            continue
        paired = pd.DataFrame({"ind": indicator, "val": wide[col]}).dropna()
        if len(paired) < 30 or paired["val"].nunique() < 3:
            continue
        r = float(paired["ind"].corr(paired["val"]))
        rows.append({
            "against": vocab.LOINC.get(col, {}).get("display", col),
            "correlation": round(r, 3),
            "abs": abs(r),
        })
    out = pd.DataFrame(rows).sort_values("abs", ascending=False, ignore_index=True)
    return out.drop(columns="abs") if len(out) else out


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def impute(wide: pd.DataFrame, strategy: str = "median", **kwargs) -> pd.DataFrame:
    """Fill missing values. Returns a new frame; the input is untouched.

      drop      complete-case analysis. The baseline every method must beat.
      mean      fast, shrinks variance, distorts every correlation.
      median    same, but not dragged by the outliers clean.py deliberately kept.
      knn       averages the k most similar encounters. Conditions on the other
                measurements, so it can exploit MAR structure.
      iterative MICE-style: model each column from the others, round-robin,
                repeat. Strongest on MAR, and the slowest.

    ON THE ConvergenceWarning
        `IterativeImputer` sometimes reports "early stopping criterion not
        reached" on smaller cohorts. It is cosmetic here: the imputed values
        are identical at max_iter 10, 20, 25 and 50 (BMI RMSE 0.719 either way
        at n=300, 1.535 at n=400), so the estimates have stabilised well before
        the tolerance test fires. Raising max_iter silences the warning without
        changing a single number, which is why it is left alone rather than
        tuned to hide a message.

    NOT MULTIPLE IMPUTATION
        sample_posterior=False, so this is a single point estimate. Proper MI
        draws m datasets and pools with Rubin's rules, which is what propagates
        imputation uncertainty into the final standard errors. Single
        imputation understates them, and any downstream confidence interval
        built on this output is too narrow.
    """
    if strategy == "drop":
        return wide.dropna()

    columns, index = wide.columns, wide.index

    if strategy in ("mean", "median"):
        imputer = SimpleImputer(strategy=strategy)
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=kwargs.get("n_neighbors", 5), weights="distance")
    elif strategy == "iterative":
        imputer = IterativeImputer(
            max_iter=kwargs.get("max_iter", 10), random_state=0, sample_posterior=False,
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    filled = imputer.fit_transform(wide)
    return pd.DataFrame(filled, columns=columns, index=index)


# ---------------------------------------------------------------------------
# Scoring against ground truth
# ---------------------------------------------------------------------------


def truth_wide(clean_observations: pd.DataFrame) -> pd.DataFrame:
    """The same wide layout, built from the pre-corruption tables."""
    wide = clean_observations.pivot_table(
        index="encounter_id", columns="loinc", values="value", aggfunc="first"
    )
    wide.columns = [str(c) for c in wide.columns]
    return wide


def score_imputation(observed: pd.DataFrame, imputed: pd.DataFrame,
                     truth: pd.DataFrame, column: str) -> dict[str, float] | None:
    """RMSE and bias over the cells that were actually missing.

    Bias is reported alongside error because they fail differently. A method
    can have low RMSE and still be systematically low -- which is exactly what
    MNAR produces, and exactly what a single error number hides.
    """
    mask = observed[column].isna()
    shared = observed.index.intersection(truth.index)
    mask = mask.reindex(shared).fillna(False)
    if not mask.any():
        return None

    predicted = imputed.loc[shared, column][mask]
    actual = truth.loc[shared, column][mask]
    paired = pd.DataFrame({"p": predicted, "a": actual}).dropna()
    if paired.empty:
        return None

    error = paired["p"] - paired["a"]
    return {
        "n_imputed": int(len(paired)),
        "rmse": round(float(np.sqrt((error ** 2).mean())), 3),
        "mae": round(float(error.abs().mean()), 3),
        "bias": round(float(error.mean()), 3),
        "true_mean": round(float(paired["a"].mean()), 3),
        "imputed_mean": round(float(paired["p"].mean()), 3),
    }


def compare_strategies(observed: pd.DataFrame, truth: pd.DataFrame,
                       columns: list[str] | None = None,
                       strategies: list[str] | None = None) -> pd.DataFrame:
    """Score every strategy on every column that has missing values.

    This table is the point of the whole module: it shows that the best
    strategy depends on the mechanism, and that none of them rescues MNAR.
    """
    strategies = strategies or ["mean", "median", "knn", "iterative"]

    # Covariates joined by attach_covariates (age, and anything else patient
    # level) are FEATURES, not targets: they help the imputer condition on the
    # MAR mechanism, but they have no ground-truth column to be scored against.
    # Impute over everything, score only what truth covers.
    columns = columns or [
        c for c in observed.columns
        if observed[c].isna().any() and c in truth.columns
    ]

    filled = {s: impute(observed, s) for s in strategies}

    rows = []
    for col in columns:
        for strategy in strategies:
            scored = score_imputation(observed, filled[strategy], truth, col)
            if scored is None:
                continue
            rows.append({
                "observation": vocab.LOINC.get(col, {}).get("display", col),
                "loinc": col,
                "strategy": strategy,
                **scored,
            })
    return pd.DataFrame(rows)


def missingness_report(wide: pd.DataFrame, mechanisms: dict[str, str]) -> Report:
    report = Report()
    report["encounters"] = len(wide)
    report["complete_rows"] = int(wide.notna().all(axis=1).sum())
    report["complete_case_loss"] = (
        f"{100 * (1 - wide.notna().all(axis=1).mean()):.1f}% of rows would be dropped"
    )
    report["cells_missing"] = int(wide.isna().sum().sum())
    report["declared_mechanisms"] = {
        vocab.LOINC.get(code, {}).get("display", code): mech
        for code, mech in mechanisms.items()
    }
    return report
