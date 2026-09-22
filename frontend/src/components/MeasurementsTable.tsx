import type { Lesion } from "@/lib/types";

const COLUMNS: { key: keyof Lesion; label: string; unit?: string; fmt?: (v: number) => string }[] = [
  { key: "id", label: "Lesion" },
  { key: "conf", label: "Confidence", fmt: (v) => v.toFixed(3) },
  { key: "max_diameter_px", label: "Max diameter", unit: "px", fmt: (v) => v.toFixed(1) },
  { key: "area_px", label: "Area", unit: "px²", fmt: (v) => v.toLocaleString() },
  { key: "circularity", label: "Circularity", fmt: (v) => v.toFixed(3) },
  { key: "solidity", label: "Solidity", fmt: (v) => v.toFixed(3) },
  { key: "echo_mean", label: "Mean echo", unit: "0–255", fmt: (v) => v.toFixed(1) },
  { key: "echo_sd", label: "Echo SD", fmt: (v) => v.toFixed(1) },
];

/** The heading and the "all values in pixels" note live in the <Section> that
 *  wraps this on the page, so they are not repeated here. */
export function MeasurementsTable({ lesions }: { lesions: Lesion[] }) {
  return (
    <>
      {/* Wide table scrolls inside its own box; the page never scrolls sideways. */}
      <div className="overflow-x-auto rounded-xl border border-white/10">
        <table className="w-full min-w-[720px] border-collapse text-sm">
          <thead>
            <tr className="bg-white/[0.04] text-left">
              {COLUMNS.map((c) => (
                <th key={c.key} className="px-3 py-2 font-medium text-white/70">
                  {c.label}
                  {c.unit && <span className="ml-1 text-xs font-normal text-white/35">({c.unit})</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {lesions.length === 0 ? (
              <tr>
                <td colSpan={COLUMNS.length} className="px-3 py-6 text-center text-white/40">
                  No measurements — run an image to populate this table.
                </td>
              </tr>
            ) : (
              lesions.map((l) => (
                <tr key={l.id} className="border-t border-white/[0.07] hover:bg-white/[0.03]">
                  {COLUMNS.map((c) => (
                    <td key={c.key} className="px-3 py-2 tabular-nums text-white/85">
                      {c.fmt ? c.fmt(l[c.key] as number) : String(l[c.key])}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <p className="mt-2 text-xs leading-relaxed text-white/40">
        Distances and areas are in image pixels. Conversion to mm or cm² requires the scanner&apos;s
        scale bar or DICOM pixel spacing, which this prototype does not read.
      </p>
    </>
  );
}
