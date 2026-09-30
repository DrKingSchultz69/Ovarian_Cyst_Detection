"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { MeasurementsTable } from "@/components/MeasurementsTable";
import { Preprocessing } from "@/components/Preprocessing";
import { ResultPanel } from "@/components/ResultPanel";
import { Uploader } from "@/components/Uploader";
import { Callout, PageHeader, Section } from "@/components/Ui";
import { predict, usingMock } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";

/** Segmentation — T4 (segmentation, restoration) and T5 (CNN), visualised per T6.
 *
 *  The burn-in inpainting in the preprocessing panel is the restoration half of
 *  T4; the YOLOv11n-seg masks are both the segmentation half and T5's CNN.
 *  Enhancement and compression live on /imaging, SIFT and RANSAC on
 *  and /registration.
 */
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [conf, setConf] = useState(0.25);
  const [debug, setDebug] = useState(true);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // Derived during render rather than pushed into state from an effect: the
  // URL is a pure function of the file, so state would just be a second copy
  // that renders one frame late. Object URLs leak unless revoked, so the
  // effect below revokes each one when it is replaced or the page unmounts.
  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );
  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const run = useCallback(
    async (f: File, c: number) => {
      setLoading(true);
      setResult(await predict(f, c, debug));
      setLoading(false);
    },
    [debug],
  );

  const onFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    // The effect below picks it up -- running here too would double-fire.
  }, []);

  // Any change to the image, the threshold or the debug flag re-runs inference.
  // Debounced so dragging the slider fires once on settle, not once per pixel;
  // driven by state rather than mouseup so keyboard and click both work.
  useEffect(() => {
    if (!file) return;
    const timer = setTimeout(() => void run(file, conf), 300);
    return () => clearTimeout(timer);
  }, [file, conf, run]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-6">
      <PageHeader title="Segmentation" topics={["T4", "T5", "T6"]}>
        YOLOv11n-seg instance masks over ultrasound, with scanner burn-in removed before
        inference. The burn-in inpainting is T4&rsquo;s <em>restoration</em>; the masks are its{" "}
        <em>segmentation</em> and T5&rsquo;s <em>CNN</em>.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        {/* Left: input and threshold */}
        <div className="space-y-4">
          <Uploader onFile={onFile} fileName={file?.name ?? null} disabled={loading} />

          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
            <div className="flex items-baseline justify-between">
              <label htmlFor="conf" className="text-sm font-medium text-white/85">
                Confidence threshold
              </label>
              <span className="text-sm tabular-nums text-sky-300">{conf.toFixed(2)}</span>
            </div>
            <input
              id="conf"
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={conf}
              onChange={(e) => setConf(Number(e.target.value))}
              aria-describedby="conf-hint"
              // Deliberately not disabled while loading: the debounce already
              // collapses rapid changes, and locking the control mid-drag
              // makes the slider feel broken.
              className="mt-3 w-full cursor-pointer accent-sky-400"
            />
            <p id="conf-hint" className="mt-2 text-xs text-white/40">
              Lower catches more and risks false positives. Higher is stricter.
              {file && <span className="ml-1 text-sky-300/70">Re-runs automatically.</span>}
            </p>

            <label className="mt-4 flex items-center gap-2 text-sm text-white/70">
              <input
                type="checkbox"
                checked={debug}
                onChange={(e) => setDebug(e.target.checked)}
                className="accent-sky-400"
              />
              Return preprocessing panels
            </label>
            {debug && (
              <p className="mt-1.5 text-[11px] leading-snug text-white/35">
                Adds ~1.4 MB to every response, and the threshold slider re-runs inference on
                each change. Turn it off when tuning the threshold.
              </p>
            )}
          </div>

          {usingMock && (
            <Callout tone="info">
              No <code className="text-sky-200">NEXT_PUBLIC_API_URL</code> set, so every upload
              returns fixed mock data. Name a file with &ldquo;empty&rdquo; to preview the
              no-detection state. The other three pages have no mock and need a real backend.
            </Callout>
          )}
        </div>

        {/* Right: annotated prediction */}
        <ResultPanel result={result} previewUrl={previewUrl} loading={loading} />
      </div>

      <Section title="Measurements" topic="T6" subtitle="All values in image pixels.">
        <MeasurementsTable lesions={result?.ok ? result.lesions : []} />
      </Section>

      <Section
        title="Preprocessing"
        topic="T4"
        subtitle="Burn-in removal — the restoration stage, shown as the three frames it passes through."
      >
        <Preprocessing debug={result?.ok ? result.debug : undefined} />
      </Section>
    </main>
  );
}
