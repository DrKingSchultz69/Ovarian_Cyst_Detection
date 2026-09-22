"use client";

import { useState } from "react";
import type { PredictDebug } from "@/lib/types";

const PANELS: { key: keyof PredictDebug; label: string; hint: string }[] = [
  { key: "raw_png", label: "1 · Raw input", hint: "as uploaded" },
  { key: "burnin_mask_png", label: "2 · Burn-in mask", hint: "caliper marks and scale ticks" },
  { key: "inpainted_png", label: "3 · Inpainted", hint: "what the model actually sees" },
];

export function Preprocessing({ debug }: { debug?: PredictDebug }) {
  const [open, setOpen] = useState(false);

  // Heading and topic badge come from the <Section> wrapping this on the page.
  return (
    <div className="overflow-hidden rounded-xl border border-white/10">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-white/85 transition hover:bg-white/[0.03]"
      >
        <span>Show the three stages</span>
        <span className={`text-white/40 transition-transform ${open ? "rotate-180" : ""}`}>▾</span>
      </button>

      {open && (
        <div className="border-t border-white/10 px-4 pb-4 pt-3">
          {!debug ? (
            <p className="py-4 text-sm text-white/40">
              Run an image with debug enabled to see the three stages.
            </p>
          ) : (
            <>
              <div className="grid gap-3 sm:grid-cols-3">
                {PANELS.map((p) => (
                  <figure key={p.key}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${debug[p.key]}`}
                      alt={p.label}
                      className="w-full rounded-lg border border-white/10 bg-black"
                    />
                    <figcaption className="mt-1.5 text-xs text-white/60">
                      {p.label}
                      <span className="ml-1 text-white/35">· {p.hint}</span>
                    </figcaption>
                  </figure>
                ))}
              </div>
              <p className="mt-3 text-xs leading-relaxed text-white/40">
                Sonographer calipers sit inside the lesion. Left in place, the model learns
                &ldquo;lesion = wherever the calipers are&rdquo; and fails on unmarked images, so they
                are detected by saturation and brightness, then Telea-inpainted before inference.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
