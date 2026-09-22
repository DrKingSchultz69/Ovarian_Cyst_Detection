"use client";

import { useEffect, useState } from "react";

import { BarChart, Histogram, LineChart, SERIES, histogramOf } from "@/components/Charts";
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
import {
  analysisAvailable,
  compressImage,
  enhanceImage,
  phantomAsFile,
  pngSrc,
} from "@/lib/analysisApi";
import { useAsync } from "@/lib/useAsync";
import type {
  CompressPoint,
  CompressResponse,
  DriftPoint,
  EnhancePanel,
  EnhanceResponse,
} from "@/lib/analysisTypes";

type Roi = [number, number, number, number];

/** T4: enhancement, restoration, segmentation and compression.
 *
 *  Segmentation and restoration already live on the home page (YOLO masks and
 *  burn-in inpainting). This page adds the two that were missing, and both are
 *  presented with the measurement next to the picture -- an enhancement demo
 *  without metrics is just a filter gallery.
 */
export default function ImagingPage() {
  // null source means "use the phantom", which the hook fetches for us. An
  // uploaded file replaces it and carries no ROI, because we do not know where
  // the lesion is in an image we did not generate.
  const [upload, setUpload] = useState<{ file: File; name: string } | null>(null);
  const [codec, setCodec] = useState("jpeg");
  const [phantomSeed, setPhantomSeed] = useState(0);

  // Stage one: get an image, either the uploaded one or a fresh phantom.
  const subject = useAsync(
    async (): Promise<{ file: File; roi: Roi | null; name: string }> => {
      if (upload) return { file: upload.file, roi: null, name: upload.name };
      const { file, bbox } = await phantomAsFile(phantomSeed);
      return { file, roi: bbox, name: "phantom" };
    },
    [upload, phantomSeed],
    { enabled: analysisAvailable },
  );

  // Stage two: analyse it. Re-runs when the subject or the codec changes.
  const analysis = useAsync(
    async () => {
      if (!subject.data) return null;
      const { file, roi } = subject.data;
      const [enhanced, compressed] = await Promise.all([
        enhanceImage(file, roi ? { roi } : {}),
        compressImage(file, { codec, ...(roi ? { roi } : {}) }),
      ]);
      return { enhanced, compressed };
    },
    [subject.data, codec],
    { enabled: analysisAvailable && Boolean(subject.data) },
  );

  const roi = subject.data?.roi ?? null;
  const source = subject.data?.name ?? "…";
  const loading = subject.loading || analysis.loading;
  const error = subject.error ?? analysis.error;
  const enhanceData: EnhanceResponse | null = analysis.data?.enhanced ?? null;
  const compressData: CompressResponse | null = analysis.data?.compressed ?? null;

  return (
    <main className="mx-auto max-w-6xl px-4 py-6">
      <PageHeader title="Enhancement & compression" topics={["T4", "T6"]}>
        The two halves of T4 that the segmentation page does not cover. Every filter is measured
        rather than judged by eye, and every compression setting is reported twice — once by how
        close the pixels stay, and once by how far the <em>measurements</em> move.
      </PageHeader>

      {!analysisAvailable ? (
        <Callout tone="warn" title="No backend configured">
          Set <code>NEXT_PUBLIC_API_URL</code> and rebuild. These endpoints have no mock.
        </Callout>
      ) : (
        <>
          <Panel>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => {
                  setUpload(null);
                  setPhantomSeed((s) => s + 1); // a different phantom each click
                }}
                className="rounded-lg bg-sky-500 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-sky-400"
              >
                {upload ? "Use synthetic phantom" : "New phantom"}
              </button>
              <label className="cursor-pointer rounded-lg border border-white/15 px-3 py-1.5 text-sm text-white/70 transition hover:bg-white/[0.04]">
                Upload an ultrasound
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) setUpload({ file: f, name: f.name });
                  }}
                />
              </label>

              <label className="ml-auto flex items-center gap-2 text-sm text-white/60">
                Codec
                <select
                  value={codec}
                  onChange={(e) => setCodec(e.target.value)}
                  className="rounded-lg border border-white/15 bg-white/[0.04] px-2 py-1 text-sm text-white/85"
                >
                  <option value="jpeg">JPEG</option>
                  <option value="webp">WebP</option>
                  <option value="png">PNG (lossless)</option>
                </select>
              </label>
            </div>
            <p className="mt-2 text-xs text-white/40">
              Source: <span className="text-white/60">{source}</span>
              {roi ? (
                <>
                  {" "}
                  · lesion ROI known, so contrast-to-noise and measurement drift are both
                  computed
                </>
              ) : (
                <>
                  {" "}
                  · no ROI, so CNR and measurement drift are skipped. The phantom supplies one
                  automatically.
                </>
              )}
            </p>
          </Panel>

          {loading && <Spinner label="Running filters and compression sweep…" />}
          {error && !loading && (
            <div className="mt-4">
              <ErrorBox error={error} />
            </div>
          )}

          {enhanceData && !loading && <Enhancement data={enhanceData} />}
          {compressData && !loading && <Compression data={compressData} />}
        </>
      )}
    </main>
  );
}

// ===========================================================================
// Enhancement
// ===========================================================================

function Enhancement({ data }: { data: EnhanceResponse }) {
  // Panel 0 is always the unprocessed original; the rest are the filters.
  const rest = data.panels.slice(1);

  // Lower speckle index is better; everything else is read in context.
  const bestSpeckle = rest.reduce((a, b) =>
    a.metrics.speckle_index <= b.metrics.speckle_index ? a : b,
  );

  return (
    <Section
      title="Enhancement"
      topic="T4"
      subtitle={
        <>
          Ultrasound speckle is <strong>multiplicative</strong> coherent interference, not
          additive noise — so filters derived under a Gaussian assumption solve the wrong
          problem. The non-local-means panel runs in the log domain, where{" "}
          <code>log(g·n) = log g + log n</code> makes the assumption true again.
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {data.panels.map((p) => (
          <PanelCard key={p.name} panel={p} best={p.name === bestSpeckle.name} roi={data.roi} />
        ))}
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Panel>
          <BarChart
            caption="Speckle index (σ/μ) — lower is less speckle"
            highlight={bestSpeckle.label}
            data={data.panels.map((p) => ({
              label: p.label,
              value: p.metrics.speckle_index,
              colour: p.name === "original" ? SERIES[3] : undefined,
            }))}
          />
        </Panel>
        <Panel>
          <BarChart
            caption="Sharpness (Laplacian variance) — rises with detail AND with noise"
            data={data.panels.map((p) => ({
              label: p.label,
              value: p.metrics.sharpness,
              colour: p.name === "original" ? SERIES[3] : SERIES[2],
            }))}
          />
        </Panel>
        <Panel>
          <BarChart
            caption="Entropy (bits) — information content"
            data={data.panels.map((p) => ({
              label: p.label,
              value: p.metrics.entropy,
              colour: p.name === "original" ? SERIES[3] : SERIES[4],
            }))}
          />
        </Panel>
      </div>

      <div className="mt-3">
        <Callout tone="info" title="Reading these three together">
          Global histogram equalisation posts the highest sharpness and the <em>worst</em>{" "}
          speckle index — it amplified the noise, which is what sharpness measures when there is
          nothing else to sharpen. CLAHE keeps the highest entropy because clipping the tile
          histograms stops it flattening the anechoic interior. The unsharp mask raises speckle
          too, which is why it belongs <em>after</em> a speckle filter and never before.
        </Callout>
      </div>

      <div className="mt-4 overflow-x-auto rounded-xl border border-white/10">
        <table className="w-full min-w-[720px] border-collapse text-sm">
          <thead>
            <tr className="bg-white/[0.04] text-left">
              <th className="px-3 py-2 font-medium text-white/70">Method</th>
              <th className="px-3 py-2 font-medium text-white/70">Group</th>
              {["mean", "std", "sharpness", "speckle_index", "entropy", "cnr"].map((k) => (
                <th key={k} className="px-3 py-2 text-right font-medium text-white/70">
                  {k.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.panels.map((p) => (
              <tr key={p.name} className="border-t border-white/[0.07] hover:bg-white/[0.03]">
                <td className="px-3 py-2 text-white/85">{p.label}</td>
                <td className="px-3 py-2 text-white/45">{p.group}</td>
                {(["mean", "std", "sharpness", "speckle_index", "entropy", "cnr"] as const).map(
                  (k) => (
                    <td key={k} className="px-3 py-2 text-right tabular-nums text-white/80">
                      {p.metrics[k] === undefined ? "—" : p.metrics[k]!.toFixed(k === "speckle_index" ? 4 : 2)}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <dl className="mt-3 grid gap-1.5 text-xs text-white/40 sm:grid-cols-2">
        {Object.entries(data.metric_notes).map(([k, v]) => (
          <div key={k}>
            <dt className="inline font-medium text-white/55">{k.replace(/_/g, " ")}: </dt>
            <dd className="inline">{v}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

/** One filter result: the image, its histogram, and the numbers. */
function PanelCard({
  panel,
  best,
  roi,
}: {
  panel: EnhancePanel;
  best: boolean;
  roi: [number, number, number, number] | null;
}) {
  const [counts, setCounts] = useState<number[]>([]);

  useEffect(() => {
    let cancelled = false;
    void histogramOf(panel.png)
      .then((c) => !cancelled && setCounts(c))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [panel.png]);

  return (
    <figure
      className={`overflow-hidden rounded-xl border bg-white/[0.02] ${
        best ? "border-emerald-400/40" : "border-white/10"
      }`}
    >
      <div className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={pngSrc(panel.png)} alt={panel.label} className="w-full bg-black" />
        {best && (
          <span className="absolute right-2 top-2 rounded-full bg-emerald-400/90 px-2 py-0.5 text-[10px] font-semibold text-emerald-950">
            lowest speckle
          </span>
        )}
      </div>

      <figcaption className="p-3">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-xs font-medium text-white/80">{panel.label}</span>
          <span className="rounded bg-white/[0.07] px-1.5 py-0.5 text-[10px] text-white/45">
            {panel.group}
          </span>
        </div>
        {panel.note && (
          <p className="mt-1 text-[11px] leading-snug text-white/40">{panel.note}</p>
        )}

        <div className="mt-2">
          <Histogram
            counts={counts}
            caption="Intensity histogram, 0–255"
            colour={best ? SERIES[1] : SERIES[0]}
          />
        </div>

        <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] tabular-nums text-white/50">
          <span>speckle {panel.metrics.speckle_index.toFixed(4)}</span>
          <span>entropy {panel.metrics.entropy.toFixed(2)}</span>
          <span>sharp {panel.metrics.sharpness.toFixed(0)}</span>
          {roi && panel.metrics.cnr !== undefined && <span>cnr {panel.metrics.cnr.toFixed(3)}</span>}
        </div>
      </figcaption>
    </figure>
  );
}

// ===========================================================================
// Compression
// ===========================================================================

function Compression({ data }: { data: CompressResponse }) {
  const lossy = data.sweep.filter((p) => !p.lossless);

  const driftColumns: Column<DriftPoint>[] = [
    { key: "quality", label: "Quality", align: "right" },
    { key: "ssim", label: "SSIM", align: "right", render: (r) => r.ssim.toFixed(5) },
    {
      key: "psnr_db",
      label: "PSNR",
      unit: "dB",
      align: "right",
      render: (r) => (r.psnr_db === null ? "lossless" : r.psnr_db.toFixed(2)),
    },
    { key: "area_px", label: "Lesion area", unit: "px", align: "right" },
    {
      key: "area_drift_pct",
      label: "Area drift",
      unit: "%",
      align: "right",
      render: (r) => (
        <span className={Math.abs(r.area_drift_pct) > 2 ? "text-rose-300" : "text-white/80"}>
          {r.area_drift_pct > 0 ? "+" : ""}
          {r.area_drift_pct.toFixed(2)}
        </span>
      ),
    },
    {
      key: "echo_drift",
      label: "Echo drift",
      align: "right",
      render: (r) => `${r.echo_drift > 0 ? "+" : ""}${r.echo_drift.toFixed(2)}`,
    },
  ];

  const sweepColumns: Column<CompressPoint>[] = [
    { key: "quality", label: "Quality", align: "right" },
    { key: "kb", label: "Size", unit: "KB", align: "right" },
    { key: "ratio", label: "Ratio", align: "right", render: (r) => `${r.ratio}×` },
    { key: "bpp", label: "Bits/px", align: "right" },
    {
      key: "psnr_db",
      label: "PSNR",
      unit: "dB",
      align: "right",
      render: (r) => (r.psnr_db === null ? "∞ (lossless)" : r.psnr_db.toFixed(2)),
    },
    { key: "ssim", label: "SSIM", align: "right", render: (r) => r.ssim.toFixed(5) },
  ];

  return (
    <Section
      title="Compression"
      topic="T4"
      subtitle={
        <>
          The question is not how small it can get, but how small before it stops being
          diagnostic — and those have different answers. Lossless PNG manages only ~4× here
          because speckle is high-entropy and looks like noise to an entropy coder, which is
          exactly what it is.
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-4">
        <Stat label="Uncompressed" value={`${data.raw_kb} KB`} />
        {data.equal_size.map((r) => (
          <Stat
            key={r.codec}
            label={`${r.label} @ ~40 KB`}
            value={r.psnr_db === null ? "lossless" : `${r.psnr_db.toFixed(1)} dB`}
            hint={`${r.kb} KB · SSIM ${r.ssim.toFixed(4)}`}
            tone={r.lossless ? "good" : undefined}
          />
        ))}
      </div>

      <div className="mt-3">
        <Callout tone="info" title="Comparing codecs at equal quality compares nothing">
          &ldquo;Quality 80&rdquo; means different things to JPEG and WebP. The row above
          binary-searches each codec for the setting that lands at the same{" "}
          <strong>file size</strong>, which is the comparison that answers which one to use.
        </Callout>
      </div>

      {lossy.length > 0 && (
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Panel>
            <LineChart
              caption="Rate-distortion. Each point is one quality setting."
              xLabel="size (KB)"
              yLabel="PSNR (dB)"
              series={[
                {
                  name: data.codecs[data.codec]?.label ?? data.codec,
                  points: lossy.map((p) => ({
                    x: p.kb,
                    y: p.psnr_db ?? 0,
                    label: `q${p.quality}`,
                  })),
                },
              ]}
            />
          </Panel>
          <Panel>
            <LineChart
              caption="SSIM against size. Note it is NOT monotonic — see below."
              xLabel="size (KB)"
              yLabel="SSIM"
              series={[
                {
                  name: "SSIM",
                  colour: SERIES[2],
                  points: lossy.map((p) => ({ x: p.kb, y: p.ssim, label: `q${p.quality}` })),
                },
              ]}
            />
          </Panel>
        </div>
      )}

      {lossy.length > 0 && (
        <div className="mt-3">
          <Callout tone="warn" title="SSIM is not monotonic here, and that is a real finding">
            PSNR falls smoothly as quality drops; SSIM does not. On speckle-dominated images the
            local variance SSIM compares is itself close to random, so whether a given
            quantisation table happens to preserve one speckle realisation is largely luck. It
            is direct evidence that SSIM — built for natural images — is on shaky ground for
            ultrasound. Quote PSNR beside it, and prefer the task metric to either.
          </Callout>
        </div>
      )}

      {data.examples && (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {data.examples.map((ex) => (
            <figure key={ex.quality} className="overflow-hidden rounded-xl border border-white/10">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={pngSrc(ex.png)}
                alt={`quality ${ex.quality}`}
                className="w-full bg-black"
              />
              <figcaption className="px-3 py-2 text-xs text-white/55">
                quality {ex.quality}
                <span className="ml-2 text-white/35">
                  {ex.kb} KB · {(data.raw_kb / ex.kb).toFixed(1)}×
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      <h3 className="mt-6 text-xs font-semibold uppercase tracking-wide text-white/45">
        Rate-distortion table
      </h3>
      <div className="mt-2">
        <DataTable columns={sweepColumns} rows={data.sweep} rowKey={(r) => String(r.quality)} />
      </div>

      {data.measurement_drift && (
        <>
          <h3 className="mt-6 text-xs font-semibold uppercase tracking-wide text-white/45">
            What actually matters — measurement drift
          </h3>
          <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-white/45">
            Fidelity degrades far faster than the measurement does: SSIM falls to ~0.86 while the
            lesion area moves under a percent. That gap is a property of{" "}
            <em>this target</em> — a large, high-contrast, sharp-walled anechoic cyst, the
            easiest possible case. A small isoechoic lesion with a diffuse border sits much
            closer to the detail JPEG discards first and should be expected to drift more.
          </p>
          <div className="mt-2">
            <DataTable
              columns={driftColumns}
              rows={data.measurement_drift}
              rowKey={(r) => String(r.quality)}
              minWidth={700}
            />
          </div>
          <div className="mt-3">
            <Callout tone="warn" title="Not a recommendation">
              Lossy compression of diagnostic images is restricted in most jurisdictions, and
              where permitted the ratio is capped by modality and must be disclosed. Nothing
              here argues for using it clinically.
            </Callout>
          </div>
        </>
      )}
    </Section>
  );
}
