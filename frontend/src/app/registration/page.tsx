"use client";

import { useEffect, useState } from "react";

import { BarChart, SERIES } from "@/components/Charts";
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
import { analysisAvailable, pngSrc, registerAndSegment, registerImages, registrationDemo } from "@/lib/analysisApi";
import { useAsync } from "@/lib/useAsync";
import type {
  RegistrationDemo,
  RegistrationFit,
  RegistrationViews,
  RegistrationSegmentationResponse,
} from "@/lib/analysisTypes";

/** T5: SIFT, RANSAC and image registration.
 *
 *  The clinical framing is a follow-up scan: before a cyst can be compared
 *  against its baseline, the two frames have to share a coordinate system.
 *  The CNN on the home page answers "what is in this frame" and says nothing
 *  about the geometry between two -- these are complementary, not rival,
 *  techniques, which is why T5 lists them together.
 */
export default function RegistrationPage() {
  const [baseline, setBaseline] = useState<File | null>(null);
  const [followUp, setFollowUp] = useState<File | null>(null);
  const [model, setModel] = useState("affine");
  const [rotation, setRotation] = useState(6);

  // The rotation slider fires on every step, and the demo endpoint takes ~3.7 s.
  // Debounce so dragging queues one request on settle rather than a dozen, and
  // let useAsync discard any that a later drag supersedes.
  const [settledRotation, setSettledRotation] = useState(rotation);
  useEffect(() => {
    const timer = setTimeout(() => setSettledRotation(rotation), 350);
    return () => clearTimeout(timer);
  }, [rotation]);

  const demoState = useAsync(
    () => registrationDemo({ rotation: settledRotation }),
    [settledRotation],
    { enabled: analysisAvailable },
  );

  // Submitted pair, set on click rather than on file choice -- registering is
  // expensive and picking two files is two separate events.
  const [submitted, setSubmitted] = useState<{ a: File; b: File; model: string } | null>(null);

  const customState = useAsync(
    async () => {
      if (!submitted) return null;
      return registerImages(submitted.a, submitted.b, { model: submitted.model });
    },
    [submitted],
    { enabled: analysisAvailable && Boolean(submitted) },
  );

  const demo = demoState.data;
  const custom = customState.data;
  const [e2e, setE2e] = useState<RegistrationSegmentationResponse | null>(null);
  const [e2eLoading, setE2eLoading] = useState(false);
  const [e2eError, setE2eError] = useState<string | null>(null);
  const loading = demoState.loading || customState.loading;
  const error = demoState.error ?? customState.error;

  return (
    <main className="mx-auto max-w-6xl px-4 py-6">
      <PageHeader title="Registration — SIFT & RANSAC" topics={["T5", "T6"]}>
        A patient returns three months later. The probe is at a different angle, at a different
        depth setting, on a different day. Before any measurement can be compared, the frames
        have to be put in one coordinate system.
      </PageHeader>

      {!analysisAvailable ? (
        <Callout tone="warn" title="No backend configured">
          Set <code>NEXT_PUBLIC_API_URL</code> and rebuild.
        </Callout>
      ) : (
        <>
          <Section
            title="The pipeline"
            topic="T5"
            subtitle={
              <>
                <strong>SIFT</strong> finds distinctive points independently in each image and
                describes each as a 4×4 grid of gradient-orientation histograms — which is what
                makes it invariant to rotation and scale.{" "}
                <strong>Lowe&rsquo;s ratio test</strong> discards matches whose best candidate is
                not clearly better than the runner-up. <strong>RANSAC</strong> then fits one
                transform, ignoring the matches that disagree with the majority.
              </>
            }
          >
            <Callout tone="warn" title="Why this is genuinely hard on ultrasound">
              SIFT keypoints are extrema of a difference-of-Gaussian scale space, and speckle
              produces thousands of them. They are only repeatable if the speckle pattern is —
              and it is not, because speckle depends on the exact probe geometry. A real second
              acquisition has an independent speckle field, so the only keypoints that survive
              are those on true anatomy, and they are a minority. The demo below deliberately
              re-speckles the follow-up frame so this difficulty is real rather than assumed.
            </Callout>
          </Section>

          {loading && <Spinner label="Detecting keypoints, matching, fitting…" />}
          {error && !loading && (
            <div className="mt-4">
              <ErrorBox error={error} />
            </div>
          )}

          {demo && !loading && <Demo demo={demo} rotation={rotation} onRotation={setRotation} />}

          <Section
            title="Register your own pair"
            topic="T5"
            subtitle="Two scans of the same anatomy. Without a known transform the fit can only be judged by the inlier ratio and the overlay views, which is the situation in the clinic."
          >
            <Panel>
              <div className="grid gap-3 sm:grid-cols-3">
                <FilePick label="Baseline" file={baseline} onPick={setBaseline} />
                <FilePick label="Follow-up" file={followUp} onPick={setFollowUp} />
                <label className="flex flex-col gap-1 text-sm text-white/60">
                  Transform model
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="rounded-lg border border-white/15 bg-white/[0.04] px-2 py-1.5 text-sm text-white/85"
                  >
                    <option value="rigid">Rigid (4 DOF)</option>
                    <option value="affine">Affine (6 DOF)</option>
                    <option value="homography">Homography (8 DOF)</option>
                  </select>
                </label>
              </div>
              <button
                type="button"
                disabled={!baseline || !followUp || loading}
                onClick={() =>
                  baseline && followUp && setSubmitted({ a: baseline, b: followUp, model })
                }
                className="mt-3 rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/40"
              >
                Register
              </button>
            </Panel>

            {custom && (
              <div className="mt-4">
                {custom.ok ? (
                  <>
                    <FitStats fit={custom} />
                    {custom.views && <Views views={custom.views} />}
                  </>
                ) : (
                  <Callout tone="warn" title="Registration failed">
                    {custom.error}
                  </Callout>
                )}
              </div>
            )}
            {baseline && followUp && (
              <div className="mt-5 border-t border-white/10 pt-5">
                <h3 className="text-sm font-semibold text-white/85">Register + segment both frames</h3>
                <p className="mt-1 text-sm text-white/50">The follow-up is registered first, then the CNN measures both scans in the shared baseline frame.</p>
                <button type="button" disabled={e2eLoading} onClick={async () => { setE2eLoading(true); setE2eError(null); try { setE2e(await registerAndSegment(baseline, followUp, { model })); } catch (e) { setE2eError(e instanceof Error ? e.message : String(e)); } finally { setE2eLoading(false); } }} className="mt-3 rounded-lg border border-emerald-400/40 bg-emerald-400/10 px-4 py-2 text-sm font-medium text-emerald-200 disabled:opacity-40">{e2eLoading ? "Running…" : "Run end-to-end comparison"}</button>
                {e2eError && <div className="mt-3"><ErrorBox error={e2eError} /></div>}
                {e2e?.comparison && <div className="mt-4 grid gap-3 sm:grid-cols-2"><Stat label="Area change" value={`${e2e.comparison.area_change_pct}%`} hint="registered follow-up vs baseline" /><Stat label="Max diameter change" value={`${e2e.comparison.max_diameter_change_pct}%`} hint="registered follow-up vs baseline" /></div>}
              </div>
            )}          </Section>
        </>
      )}
    </main>
  );
}

// ---------------------------------------------------------------------------

function FilePick({
  label,
  file,
  onPick,
}: {
  label: string;
  file: File | null;
  onPick: (f: File) => void;
}) {
  return (
    <label className="flex cursor-pointer flex-col gap-1 text-sm text-white/60">
      {label}
      <span className="truncate rounded-lg border border-white/15 bg-white/[0.04] px-2 py-1.5 text-sm text-white/85">
        {file?.name ?? "Choose file…"}
      </span>
      <input
        type="file"
        accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPick(f);
        }}
      />
    </label>
  );
}

function FitStats({ fit }: { fit: RegistrationFit }) {
  const trustworthy = fit.inlier_ratio >= 0.3;
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      <Stat
        label="Keypoints"
        value={`${fit.keypoints_baseline} / ${fit.keypoints_followup}`}
        hint="baseline / follow-up"
      />
      <Stat
        label="After ratio test"
        value={fit.matches_after_ratio_test}
        hint={`${(
          (100 * fit.matches_after_ratio_test) /
          Math.max(fit.keypoints_baseline, 1)
        ).toFixed(1)}% survived`}
      />
      <Stat label="RANSAC inliers" value={fit.inliers} />
      <Stat
        label="Inlier ratio"
        value={fit.inlier_ratio.toFixed(3)}
        tone={trustworthy ? "good" : "bad"}
        hint={trustworthy ? "above the 0.3 floor" : "below 0.3 — do not trust this"}
      />
      <Stat
        label="Reprojection RMSE"
        value={fit.rmse_px === null ? "—" : `${fit.rmse_px.toFixed(3)} px`}
        hint="over inliers only"
      />
    </div>
  );
}

function Views({ views }: { views: RegistrationViews }) {
  const panels: { key: keyof RegistrationViews; label: string; hint: string }[] = [
    { key: "matches", label: "1 · Inlier matches", hint: "RANSAC-accepted correspondences" },
    { key: "warped", label: "2 · Follow-up warped", hint: "resampled onto the baseline grid" },
    {
      key: "checkerboard",
      label: "3 · Checkerboard",
      hint: "misalignment breaks structure at tile edges",
    },
    { key: "difference", label: "4 · Difference", hint: "bright means poorly aligned" },
  ];

  return (
    <div className="mt-4 grid gap-3 sm:grid-cols-2">
      {panels.map((p) => (
        <figure key={p.key} className="overflow-hidden rounded-xl border border-white/10">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={pngSrc(views[p.key])} alt={p.label} className="w-full bg-black" />
          <figcaption className="px-3 py-2 text-xs text-white/60">
            {p.label}
            <span className="ml-1 text-white/35">· {p.hint}</span>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------

function Demo({
  demo,
  rotation,
  onRotation,
}: {
  demo: RegistrationDemo;
  rotation: number;
  onRotation: (v: number) => void;
}) {
  const columns: Column<RegistrationFit>[] = [
    {
      key: "model",
      label: "Model",
      render: (r) => (
        <span className="font-medium text-white/85">
          {r.model}
          <span className="ml-1.5 text-[11px] text-white/35">
            {{ rigid: "4 DOF", affine: "6 DOF", homography: "8 DOF" }[r.model]}
          </span>
        </span>
      ),
    },
    { key: "inliers", label: "Inliers", align: "right" },
    {
      key: "inlier_ratio",
      label: "Inlier ratio",
      align: "right",
      render: (r) => r.inlier_ratio.toFixed(3),
    },
    {
      key: "rmse_px",
      label: "Reprojection RMSE",
      unit: "px",
      align: "right",
      render: (r) => (r.rmse_px === null ? "—" : r.rmse_px.toFixed(3)),
    },
    {
      key: "corner_error_px",
      label: "TRUE corner error",
      unit: "px",
      align: "right",
      render: (r) =>
        r.corner_error_px === undefined ? (
          "—"
        ) : (
          <span
            className={
              r.corner_error_px < 1 ? "font-semibold text-emerald-300" : "font-semibold text-rose-300"
            }
          >
            {r.corner_error_px.toFixed(3)}
          </span>
        ),
    },
  ];

  const scored = demo.models.filter((m) => m.corner_error_px !== undefined);
  const bestRmse = scored.reduce((a, b) => ((a.rmse_px ?? 9e9) <= (b.rmse_px ?? 9e9) ? a : b));
  const bestTrue = scored.reduce((a, b) =>
    (a.corner_error_px ?? 9e9) <= (b.corner_error_px ?? 9e9) ? a : b,
  );
  const disagree = bestRmse.model !== bestTrue.model;

  return (
    <Section
      title="Scored against a known transform"
      topic="T5"
      subtitle={
        <>
          The phantom pair is generated by applying a transform we choose, then re-speckling the
          follow-up independently. Because the answer is known, each recovered model can be
          scored by how far it moves the image corners — a geometric error in pixels, which is
          interpretable, rather than a matrix norm, which is not.
        </>
      }
    >
      <Panel>
        <label className="flex flex-wrap items-center gap-3 text-sm text-white/60">
          Applied rotation
          <input
            type="range"
            min={-30}
            max={30}
            step={2}
            value={rotation}
            onChange={(e) => onRotation(Number(e.target.value))}
            className="w-56 cursor-pointer accent-sky-400"
          />
          <span className="tabular-nums text-sky-300">{rotation}°</span>
          <span className="text-xs text-white/35">
            Larger rotations leave fewer repeatable keypoints — watch the inlier ratio fall.
          </span>
        </label>
      </Panel>

      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <figure className="overflow-hidden rounded-xl border border-white/10">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={pngSrc(demo.baseline_png)} alt="Baseline" className="w-full bg-black" />
          <figcaption className="px-3 py-2 text-xs text-white/60">Baseline</figcaption>
        </figure>
        <figure className="overflow-hidden rounded-xl border border-white/10">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={pngSrc(demo.followup_png)} alt="Follow-up" className="w-full bg-black" />
          <figcaption className="px-3 py-2 text-xs text-white/60">
            Follow-up
            <span className="ml-1 text-white/35">
              · transformed and independently re-speckled
            </span>
          </figcaption>
        </figure>
      </div>

      <div className="mt-4">
        <DataTable
          columns={columns}
          rows={demo.models}
          rowKey={(r) => r.model}
          minWidth={620}
          highlight={(r) => r.model === bestTrue.model}
        />
      </div>

      {disagree && (
        <div className="mt-3">
          <Callout tone="warn" title="The best RMSE is not the best transform">
            <strong>{bestRmse.model}</strong> has the lowest reprojection RMSE (
            {bestRmse.rmse_px?.toFixed(3)} px) and <strong>{bestTrue.model}</strong> has the
            lowest true error ({bestTrue.corner_error_px?.toFixed(3)} px vs{" "}
            {bestRmse.corner_error_px?.toFixed(3)} px). That is overfitting made visible: eight
            degrees of freedom absorb noise the correct model cannot, so the residual a
            homography reports <em>on its own inliers</em> shrinks while the transform drifts
            further from the truth.
            <div className="mt-1.5">
              Two consequences. RMSE alone must never be used to choose a model — only a
              held-out ground truth separates them, and in the clinic there is not one. And a
              homography is the wrong <em>model</em> here regardless: two ultrasound sweeps are
              different cross-sections of a deforming 3D organ, not two views of a plane.
            </div>
          </Callout>
        </div>
      )}

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel>
          <BarChart
            caption="Reprojection RMSE — what the fit reports about itself"
            unit="px"
            data={scored.map((m) => ({
              label: m.model,
              value: m.rmse_px ?? 0,
              colour: SERIES[0],
            }))}
          />
        </Panel>
        <Panel>
          <BarChart
            caption="True corner error — measured against the known transform"
            unit="px"
            highlight={bestTrue.model}
            data={scored.map((m) => ({
              label: m.model,
              value: m.corner_error_px ?? 0,
              colour: m.model === bestTrue.model ? SERIES[1] : SERIES[5],
            }))}
          />
        </Panel>
      </div>

      {demo.views && (
        <>
          <h3 className="mt-6 text-xs font-semibold uppercase tracking-wide text-white/45">
            Affine result, four ways
          </h3>
          <Views views={demo.views} />
        </>
      )}

      <div className="mt-4">
        <Panel>
          <h4 className="text-xs font-semibold text-white/60">Transform applied (ground truth)</h4>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-white/60">
            {demo.true_transform.map((row) => row.map((v) => v.toFixed(4).padStart(10)).join("  ")).join("\n")}
          </pre>
          <p className="mt-2 text-[11px] leading-relaxed text-white/35">
            Maps baseline → follow-up. The recovered matrices map the other way, so they are
            compared against this matrix inverted.
          </p>
        </Panel>
      </div>
    </Section>
  );
}
