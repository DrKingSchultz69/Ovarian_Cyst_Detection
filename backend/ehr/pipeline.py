"""End-to-end EHR pipeline -- T1 through T3 in one pass.

    generate -> standardize -> clean -> dedupe -> impute

Order is not negotiable, and each step depends on the one before:

  standardize before clean    a range check applied to unconverted units
                              rejects every imperial height
  clean before dedupe         matching on unparsed dates blocks on the string
                              "29/06/2007", which never equals "2007-06-29"
  dedupe before impute        duplicate rows bias every column mean that
                              imputation is about to draw from

No web-framework imports, matching inference.py: the whole pipeline is
runnable and inspectable from a REPL.

    from ehr.pipeline import run
    result = run()
    print(result.reports["clean"])
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from . import clean, dedupe, generate, missing, standardize, vocab


@dataclasses.dataclass
class PipelineResult:
    """Every intermediate stage, kept rather than discarded.

    Holding the intermediates is what makes the pipeline auditable -- the
    frontend renders before/after pairs directly from these, and a reviewer
    can diff any two stages without a rerun.
    """

    raw: dict[str, pd.DataFrame]
    standardised: dict[str, pd.DataFrame]
    cleaned: dict[str, pd.DataFrame]
    deduped: dict[str, pd.DataFrame]
    wide_observed: pd.DataFrame
    wide_truth: pd.DataFrame
    imputed: pd.DataFrame
    flags: pd.DataFrame
    match_pairs: pd.DataFrame
    reports: dict[str, Any]
    manifest: generate.Manifest


def run(n_patients: int = 400, seed: int = 7,
        impute_strategy: str = "iterative") -> PipelineResult:
    """Build a cohort and take it all the way through."""
    cohort = generate.build(n_patients=n_patients, seed=seed)
    reports: dict[str, Any] = {}

    # --- T1: what the raw extract looks like ------------------------------
    reports["source"] = standardize.Report({
        "patients": len(cohort.dirty["patients"]),
        "encounters": len(cohort.dirty["encounters"]),
        "observations": len(cohort.dirty["observations"]),
        "imaging_studies": len(cohort.dirty["imaging_studies"]),
        "injected_defects": cohort.manifest.summary(),
    })

    # --- T2a: standardization --------------------------------------------
    standardised, std_reports = standardize.standardise_all(cohort.dirty)
    reports["standardize"] = std_reports

    # --- T2b: cleaning ----------------------------------------------------
    cleaned, flags, clean_report = clean.clean_all(standardised)
    reports["clean"] = clean_report

    # --- T3a: deduplication ----------------------------------------------
    deduped, pairs, dedupe_report = dedupe.dedupe_all(cleaned)
    reports["dedupe"] = dedupe_report
    reports["dedupe_evaluation"] = dedupe.evaluate(pairs, cohort.manifest.duplicate_of)

    # --- T3b: missing data ------------------------------------------------
    wide = missing.to_wide(deduped["observations"])
    wide = missing.attach_covariates(wide, deduped["encounters"], deduped["patients"])
    truth = missing.truth_wide(cohort.clean["observations"])

    reports["missing"] = missing.missingness_report(wide, cohort.manifest.missing_mechanism)
    reports["missing_profile"] = missing.profile(wide)
    reports["mechanism_diagnostics"] = {
        vocab.LOINC[code]["display"]: missing.little_style_correlation(wide, code).head(3)
        for code in (vocab.Loinc.CA125, vocab.Loinc.BMI, vocab.Loinc.HEMOGLOBIN)
        if code in wide.columns
    }
    reports["imputation_comparison"] = missing.compare_strategies(wide, truth)

    imputed = missing.impute(wide, impute_strategy)

    return PipelineResult(
        raw=cohort.dirty,
        standardised=standardised,
        cleaned=cleaned,
        deduped=deduped,
        wide_observed=wide,
        wide_truth=truth,
        imputed=imputed,
        flags=flags,
        match_pairs=pairs,
        reports=reports,
        manifest=cohort.manifest,
    )


def summarise(result: PipelineResult) -> str:
    """Human-readable run summary, for the CLI and the smoke test."""
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    section("T1  source extract")
    lines.append(str(result.reports["source"]))

    section("T2  standardization")
    for table, report in result.reports["standardize"].items():
        lines.append(f"  [{table}]")
        lines.append(str(report))

    section("T2  cleaning")
    lines.append(str(result.reports["clean"]))

    section("T3  deduplication")
    lines.append(str(result.reports["dedupe"]))
    lines.append("")
    lines.append("  scored against the generator manifest:")
    lines.append(str(result.reports["dedupe_evaluation"]))

    section("T3  missing data")
    lines.append(str(result.reports["missing"]))
    lines.append("")
    lines.append(result.reports["missing_profile"].to_string(index=False))

    section("T3  mechanism diagnostics  (what predicts a value being absent)")
    for name, table in result.reports["mechanism_diagnostics"].items():
        lines.append(f"  {name}")
        lines.append("    " + table.to_string(index=False).replace("\n", "\n    "))

    section("T3  imputation scored against ground truth")
    comparison = result.reports["imputation_comparison"]
    lines.append(comparison.to_string(index=False))

    return "\n".join(lines)


if __name__ == "__main__":
    print(summarise(run()))
