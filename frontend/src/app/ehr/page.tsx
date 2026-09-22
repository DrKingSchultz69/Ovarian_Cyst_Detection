"use client";

import { useState } from "react";

import { BarChart, ProportionBar, SERIES } from "@/components/Charts";
import {
  Callout,
  Column,
  DataTable,
  ErrorBox,
  PageHeader,
  Panel,
  Section,
  Spinner,
  Stat,
} from "@/components/Ui";
import { analysisAvailable, getEhrPipeline } from "@/lib/analysisApi";
import { useAsync } from "@/lib/useAsync";
import type {
  CleanFlag,
  EhrResponse,
  ImputationRow,
  Icd10Row,
  LoincRow,
  MatchPair,
} from "@/lib/analysisTypes";

/** T1-T3: the EHR half of the syllabus.
 *
 *  One page rather than three, because the three topics are three stages of
 *  one pipeline and splitting them would hide the dependency that matters
 *  most: standardization has to precede cleaning, and both have to precede
 *  deduplication and imputation.
 */
export default function EhrPage() {
  const [seed, setSeed] = useState(7);

  const { data, error, loading } = useAsync(
    () => getEhrPipeline({ seed, rows: 30 }),
    [seed],
    { enabled: analysisAvailable },
  );

  return (
    <main className="mx-auto max-w-6xl px-4 py-6">
      <PageHeader title="EHR pipeline" topics={["T1", "T2", "T3"]}>
        A synthetic cohort of ovarian-ultrasound patients, one per MMOTU scan, generated clean
        and then deliberately corrupted. Because every defect was recorded when it was
        injected, each repair below is <em>scored</em> against the truth it is trying to
        recover rather than asserted. No real patient data is involved.
      </PageHeader>

      {!analysisAvailable ? (
        <Callout tone="warn" title="No backend configured">
          This page needs a running API. Set <code>NEXT_PUBLIC_API_URL</code> and rebuild —{" "}
          <code>NEXT_PUBLIC_*</code> is inlined at build time, so a restart alone will not pick
          it up. Unlike the segmentation page there is no mock: a mocked pipeline would only be
          demonstrating its own fixtures.
        </Callout>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-sm text-white/60">
              Cohort seed
              <input
                type="number"
                min={0}
                max={9999}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-20 rounded-lg border border-white/15 bg-white/[0.04] px-2 py-1 text-sm tabular-nums text-white/85"
              />
            </label>
            <span className="text-xs text-white/35">
              Deterministic: the same seed always regenerates the same cohort and the same
              defects.
            </span>
          </div>

          {loading && <Spinner label="Running generate → standardize → clean → dedupe → impute…" />}
          {error && !loading && (
            <div className="mt-4">
              <ErrorBox error={error} />
            </div>
          )}
          {data && !loading && <Pipeline data={data} />}
        </>
      )}
    </main>
  );
}

// ===========================================================================

function Pipeline({ data }: { data: EhrResponse }) {
  return (
    <>
      <T1 data={data} />
      <T2 data={data} />
      <T3 data={data} />
    </>
  );
}

// ---------------------------------------------------------------------------
// T1
// ---------------------------------------------------------------------------

function T1({ data }: { data: EhrResponse }) {
  const icd10Columns: Column<Icd10Row>[] = [
    { key: "code", label: "ICD-10" },
    { key: "display", label: "Display" },
    {
      key: "local_synonyms",
      label: "Local text that maps to it",
      render: (r) => (
        <span className="text-white/55">{r.local_synonyms.join(", ") || "—"}</span>
      ),
    },
  ];

  const loincColumns: Column<LoincRow>[] = [
    { key: "code", label: "LOINC" },
    { key: "display", label: "Observation" },
    { key: "canonical_unit", label: "Canonical unit" },
    {
      key: "plausible_range",
      label: "Plausible range",
      align: "right",
      render: (r) => `${r.plausible_range[0]} – ${r.plausible_range[1]}`,
    },
    {
      key: "local_names",
      label: "Local names absorbed",
      render: (r) => <span className="text-white/55">{r.local_names.join(", ")}</span>,
    },
  ];

  return (
    <Section
      id="t1"
      title="Understanding the EHR"
      topic="T1"
      subtitle={
        <>
          An EHR is four linked things: <strong>who</strong> the patient is, <strong>when</strong>{" "}
          they were seen, <strong>what</strong> was measured, and <strong>what was found</strong>.
          Those are the four tables below. What makes them interoperable rather than just
          structured is that the contents are <em>coded</em> — and the raw extract arrives with
          local spellings instead, which is the whole of T2.
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Patient rows" value={data.source.patients} hint="registrations, not people" />
        <Stat label="Encounters" value={data.source.encounters} hint="visits" />
        <Stat label="Observations" value={data.source.observations} hint="one per measurement" />
        <Stat label="Imaging studies" value={data.source.imaging_studies} hint="one per scan" />
      </div>

      <div className="mt-3">
        <Callout tone="info" title="Defects injected on purpose">
          <div className="flex flex-wrap gap-x-5 gap-y-1">
            {Object.entries(data.source.injected_defects).map(([k, v]) => (
              <span key={k}>
                <span className="font-semibold tabular-nums">{v}</span>{" "}
                {k.replace(/_/g, " ")}
              </span>
            ))}
          </div>
          <div className="mt-1.5 text-sky-100/55">
            Recorded at injection time, which is what lets every repair below be measured.
          </div>
        </Callout>
      </div>

      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-white/45">
        Coding system 1 — ICD-10, diagnoses
      </h3>
      <div className="mt-2">
        <DataTable columns={icd10Columns} rows={data.coding_systems.icd10} rowKey={(r) => r.code} />
      </div>

      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-white/45">
        Coding system 2 — LOINC, observations
      </h3>
      <div className="mt-2">
        <DataTable
          columns={loincColumns}
          rows={data.coding_systems.loinc}
          rowKey={(r) => r.code}
          minWidth={820}
        />
      </div>
      <p className="mt-2 text-xs leading-relaxed text-white/40">{data.coding_systems.note}</p>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// T2
// ---------------------------------------------------------------------------

function T2({ data }: { data: EhrResponse }) {
  const obs = data.standardize.observations as Record<string, number | string>;
  const patients = data.standardize.patients as Record<string, number | string>;
  const studies = data.standardize.imaging_studies as Record<string, number | string>;

  const collapse = [
    {
      label: "Local test names → LOINC",
      before: Number(obs.distinct_local_test_names),
      after: data.coding_systems.loinc.length,
    },
    {
      label: "Units → canonical",
      before: Number(obs.distinct_units_before),
      after: Number(obs.distinct_units_after),
    },
    {
      label: "Sex spellings → FHIR",
      before: Number(patients.sex_distinct_before),
      after: Number(patients.sex_distinct_after),
    },
    {
      label: "Diagnosis text → ICD-10",
      before: Number(studies.distinct_diagnosis_text),
      after: Number(studies.distinct_icd10_codes),
    },
  ];

  const flagColumns: Column<CleanFlag>[] = [
    {
      key: "rule",
      label: "Rule",
      render: (r) => <span className="font-medium text-white/80">{r.rule}</span>,
    },
    { key: "value", label: "Value", align: "right", render: (r) => String(r.value ?? "—") },
    {
      key: "action",
      label: "Action",
      render: (r) => (
        <span
          className={
            r.action === "quarantined"
              ? "rounded bg-rose-400/15 px-1.5 py-0.5 text-[11px] text-rose-200"
              : "rounded bg-amber-400/15 px-1.5 py-0.5 text-[11px] text-amber-200"
          }
        >
          {r.action}
        </span>
      ),
    },
    { key: "detail", label: "Why", render: (r) => <span className="text-white/55">{r.detail}</span> },
  ];

  return (
    <Section
      id="t2"
      title="Standardization and cleaning"
      topic="T2"
      subtitle={
        <>
          Two different jobs, in a fixed order. <strong>Standardization</strong> changes
          representation and never meaning — one date, one unit, one code for one concept.{" "}
          <strong>Cleaning</strong> then decides which values are wrong. Reversing the order
          breaks it: a 40–230&nbsp;cm height check applied before unit conversion rejects every
          patient measured in inches.
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-white/45">
            Vocabulary collapse
          </h3>
          <div className="mt-3 space-y-3">
            {collapse.map((c) => (
              <div key={c.label}>
                <div className="flex items-baseline justify-between text-xs">
                  <span className="text-white/60">{c.label}</span>
                  <span className="tabular-nums">
                    <span className="text-rose-300">{c.before}</span>
                    <span className="mx-1.5 text-white/30">→</span>
                    <span className="font-semibold text-emerald-300">{c.after}</span>
                  </span>
                </div>
                <div className="mt-1 flex h-1.5 gap-0.5 overflow-hidden rounded-full">
                  <div
                    className="rounded-l-full bg-rose-400/50"
                    style={{ width: `${(c.before / (c.before + c.after)) * 100}%` }}
                  />
                  <div
                    className="rounded-r-full bg-emerald-400/70"
                    style={{ width: `${(c.after / (c.before + c.after)) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-white/35">
            Every local name mapped: <strong className="text-white/50">{obs.mapped_to_loinc}</strong>{" "}
            of {obs.rows} observations, with {String(obs.unmapped_test_names)} unmapped. Unmapped
            rows would be kept with a null code, never dropped.
          </p>
        </Panel>

        <Panel>
          <h3 className="text-xs font-semibold uppercase tracking-wide text-white/45">
            Cleaning outcomes
          </h3>
          <div className="mt-3 grid grid-cols-3 gap-2">
            <Stat label="Flags" value={data.clean.flags_raised} />
            <Stat label="Quarantined" value={data.clean.values_quarantined} tone="bad" />
            <Stat label="Flagged only" value={data.clean.values_flagged_only} tone="warn" />
          </div>
          {data.clean.by_rule && (
            <div className="mt-3">
              <BarChart
                caption="Flags raised by rule"
                data={Object.entries(data.clean.by_rule).map(([rule, n]) => ({
                  label: rule,
                  value: n,
                  colour:
                    rule === "outlier" ? SERIES[2] : rule === "plausibility" ? SERIES[5] : SERIES[0],
                }))}
              />
            </div>
          )}
          <div className="mt-2">
            <Callout tone="warn">
              Statistical outliers are <strong>flagged, never deleted</strong>. In a disease
              cohort the extreme CA-125 values are the positive cases — an IQR filter here would
              remove exactly the patients the study is about.
            </Callout>
          </div>
        </Panel>
      </div>

      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-white/45">
        Every rejection, with its reason
      </h3>
      <div className="mt-2">
        <DataTable
          columns={flagColumns}
          rows={data.flags}
          rowKey={(r, i) => `${r.rule}-${r.row}-${i}`}
          empty="No flags raised."
          minWidth={700}
        />
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// T3
// ---------------------------------------------------------------------------

function T3({ data }: { data: EhrResponse }) {
  const ev = data.dedupe_evaluation;

  const pairColumns: Column<MatchPair>[] = [
    { key: "left_id", label: "MRN A" },
    { key: "right_id", label: "MRN B" },
    { key: "left_name", label: "Name A" },
    { key: "right_name", label: "Name B" },
    { key: "birth_date", label: "Shared DOB" },
    {
      key: "score",
      label: "Score",
      align: "right",
      render: (r) => <span className="font-semibold text-sky-300">{r.score.toFixed(3)}</span>,
    },
  ];

  // Group the imputation comparison so each observation gets its own chart.
  const byObservation = data.imputation_comparison.reduce<Record<string, ImputationRow[]>>(
    (acc, row) => {
      (acc[row.observation] ??= []).push(row);
      return acc;
    },
    {},
  );

  // The three declared mechanisms are the interesting ones; show those first.
  const featured = ["Body mass index", "Cancer Ag 125", "Hemoglobin"].filter(
    (k) => k in byObservation,
  );

  return (
    <Section
      id="t3"
      title="Redundant data removal and missing data"
      topic="T3"
      subtitle={
        <>
          Two problems that both look like &ldquo;bad rows&rdquo; and need opposite treatments:
          duplicates must be <em>merged</em>, absences must be <em>modelled</em>. Getting either
          wrong destroys clinical history.
        </>
      }
    >
      {/* --- dedupe ----------------------------------------------------- */}
      <h3 className="text-xs font-semibold uppercase tracking-wide text-white/45">
        Record linkage
      </h3>

      <div className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Precision" value={ev.precision.toFixed(3)} tone={ev.precision === 1 ? "good" : "warn"} hint="no wrong merges" />
        <Stat label="Recall" value={ev.recall.toFixed(3)} tone={ev.recall >= 0.95 ? "good" : "warn"} hint="found / injected" />
        <Stat label="F1" value={ev.f1.toFixed(3)} tone="good" />
        <Stat
          label="Comparisons"
          value={String(data.dedupe.comparisons_after_blocking ?? "—")}
          hint={`of ${data.dedupe.all_pairs_would_be} possible pairs`}
        />
      </div>

      <div className="mt-3 grid gap-4 lg:grid-cols-2">
        <Panel>
          <BarChart
            caption="Linkage outcomes against the injected ground truth"
            data={[
              { label: "True positives", value: ev.true_positives, colour: SERIES[1] },
              { label: "False positives", value: ev.false_positives, colour: SERIES[5] },
              { label: "False negatives", value: ev.false_negatives, colour: SERIES[2] },
            ]}
          />
          <Callout tone="info" title="Why both error types matter">
            A <strong>missed</strong> duplicate splits one person across two charts, so an
            allergy recorded on one is invisible on the other. A <strong>false</strong> match
            merges two people, and one patient inherits another&rsquo;s diagnosis. The second is
            the more dangerous failure, which is why the threshold is set where precision stays
            at 1.0.
          </Callout>
        </Panel>

        <Panel>
          <h4 className="text-xs font-semibold text-white/60">Blocking</h4>
          <p className="mt-1.5 text-xs leading-relaxed text-white/45">
            Comparing all pairs is O(n²) — {String(data.dedupe.all_pairs_would_be)} comparisons
            for this cohort, and 125 billion for a real one of 500k. Blocking on exact birth date
            cuts it to {String(data.dedupe.comparisons_after_blocking)} (
            {String(data.dedupe.reduction)} fewer). The cost is a missed match whenever the
            blocking key itself is wrong, so it has to be a field that is rarely mistyped and
            never legitimately changes.
          </p>
          <div className="mt-3 space-y-2">
            <ProportionBar
              label="Exact duplicates removed"
              pct={100}
              right={String(data.dedupe.exact_duplicates_removed)}
              tone="rose"
            />
            <ProportionBar
              label="Clusters merged (fuzzy)"
              pct={100}
              right={String(data.dedupe.clusters_merged)}
              tone="amber"
            />
            <ProportionBar
              label="Rows surviving"
              pct={100}
              right={String(data.dedupe.rows_out)}
              tone="emerald"
            />
          </div>
          <Callout tone="warn" title="The case that is not a duplicate">
            <span>
              One person attending twice shares an MRN and differs only in encounter. It is{" "}
              <strong>not</strong> redundant, and the pipeline deliberately leaves those rows
              alone — a deduplicator that collapses them is deleting visit history.
            </span>
          </Callout>
        </Panel>
      </div>

      <div className="mt-3">
        <DataTable
          columns={pairColumns}
          rows={data.match_pairs}
          rowKey={(r) => `${r.left_id}-${r.right_id}`}
          empty="No candidate pairs above threshold."
          minWidth={760}
        />
      </div>

      {/* --- missing ------------------------------------------------------ */}
      <h3 className="mt-7 text-xs font-semibold uppercase tracking-wide text-white/45">
        Missing data
      </h3>

      <div className="mt-2 grid gap-3 sm:grid-cols-3">
        <Stat label="Cells missing" value={data.missing.cells_missing} />
        <Stat label="Complete rows" value={data.missing.complete_rows} hint={`of ${data.missing.encounters}`} />
        <Stat
          label="Complete-case loss"
          value={data.missing.complete_case_loss.split("%")[0] + "%"}
          tone="bad"
          hint="dropped by listwise deletion"
        />
      </div>

      <div className="mt-3">
        <Callout tone="warn" title="Why listwise deletion is not the safe default">
          Dropping any row with a gap discards{" "}
          <strong>{data.missing.complete_case_loss}</strong>. Under MCAR that is merely wasteful;
          under MAR or MNAR it is also biased, because the rows that survive are not a random
          sample of the cohort.
        </Callout>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel>
          <h4 className="mb-3 text-xs font-semibold text-white/60">Missingness per observation</h4>
          <div className="space-y-2">
            {data.missing_profile.map((r) => (
              <ProportionBar
                key={r.loinc}
                label={r.display}
                pct={r.missing_pct}
                right={`${r.missing_pct.toFixed(1)}%`}
                tone={r.missing_pct > 20 ? "rose" : r.missing_pct > 8 ? "amber" : "sky"}
              />
            ))}
          </div>
        </Panel>

        <Panel>
          <h4 className="text-xs font-semibold text-white/60">
            Mechanism diagnostics — what predicts a value being absent
          </h4>
          <p className="mt-1.5 text-[11px] leading-relaxed text-white/40">
            If missingness correlates with an observed variable, the data is not MCAR. This can
            refute MCAR and separate it from MAR. It <em>cannot</em> separate MAR from MNAR —
            the evidence that would is exactly the data that is absent. The mechanisms below are
            known only because this cohort was generated.
          </p>
          <div className="mt-3 space-y-4">
            {Object.entries(data.mechanism_diagnostics).map(([name, rows]) => {
              const declared = data.missing.declared_mechanisms[name] ?? "—";
              return (
                <div key={name}>
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="text-xs font-medium text-white/70">{name}</span>
                    <span className="rounded bg-white/[0.07] px-1.5 py-0.5 text-[10px] text-white/50">
                      declared {declared}
                    </span>
                  </div>
                  <div className="mt-1.5">
                    <BarChart
                      caption=""
                      diverging
                      data={rows.map((r) => ({
                        label: r.against,
                        value: r.correlation,
                        colour: Math.abs(r.correlation) > 0.3 ? SERIES[2] : SERIES[0],
                      }))}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-white/40">
            Read the three together: CA-125 correlates strongly with age (MAR detected), BMI
            correlates with weight because weight is a proxy for the hidden BMI value (MNAR
            leaking), and haemoglobin correlates with nothing above noise (consistent with
            MCAR).
          </p>
        </Panel>
      </div>

      {/* --- imputation --------------------------------------------------- */}
      <h4 className="mt-6 text-xs font-semibold uppercase tracking-wide text-white/45">
        Imputation, scored against the deleted values
      </h4>
      <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-white/45">
        Bias is shown next to error because they fail differently. A method can have a
        respectable RMSE and still be systematically low — which is precisely what MNAR
        produces, and precisely what a single error number hides.
      </p>

      <div className="mt-3 grid gap-4 lg:grid-cols-3">
        {featured.map((name) => {
          const rows = byObservation[name];
          const best = rows.reduce((a, b) => (Math.abs(a.bias) <= Math.abs(b.bias) ? a : b));
          return (
            <Panel key={name}>
              <div className="flex flex-wrap items-baseline gap-2">
                <span className="text-xs font-medium text-white/75">{name}</span>
                <span className="rounded bg-white/[0.07] px-1.5 py-0.5 text-[10px] text-white/50">
                  {data.missing.declared_mechanisms[name] ?? "—"}
                </span>
              </div>
              <div className="mt-3">
                <BarChart
                  caption="Bias — imputed mean minus true mean"
                  diverging
                  highlight={best.strategy}
                  data={rows.map((r) => ({
                    label: r.strategy,
                    value: r.bias,
                    hint: `RMSE ${r.rmse}, MAE ${r.mae}, n=${r.n_imputed}`,
                  }))}
                />
              </div>
              <div className="mt-2">
                <BarChart
                  caption="RMSE"
                  highlight={rows.reduce((a, b) => (a.rmse <= b.rmse ? a : b)).strategy}
                  data={rows.map((r) => ({ label: r.strategy, value: r.rmse }))}
                />
              </div>
            </Panel>
          );
        })}
      </div>

      <div className="mt-3">
        <Callout tone="info" title="The MNAR result is not what it looks like">
          On BMI — declared MNAR — iterative imputation almost eliminates the bias while mean and
          median stay ~6 units low. That is <strong>not</strong> MICE solving MNAR. BMI is
          derivable from height and weight, both of which are present, so the information was
          never actually missing; MICE finds the deterministic relationship and reconstructs it.
          For a genuinely unobservable MNAR quantity no imputation recovers the distribution,
          and the honest options are to model the mechanism explicitly or to declare the
          limitation.
        </Callout>
      </div>

      <div className="mt-4">
        <DataTable
          columns={[
            { key: "observation", label: "Observation" },
            { key: "strategy", label: "Strategy" },
            { key: "n_imputed", label: "n", align: "right" },
            { key: "rmse", label: "RMSE", align: "right" },
            { key: "mae", label: "MAE", align: "right" },
            { key: "bias", label: "Bias", align: "right" },
            { key: "true_mean", label: "True mean", align: "right" },
            { key: "imputed_mean", label: "Imputed mean", align: "right" },
          ] as Column<ImputationRow>[]}
          rows={data.imputation_comparison}
          rowKey={(r) => `${r.loinc}-${r.strategy}`}
          minWidth={820}
        />
      </div>
    </Section>
  );
}
