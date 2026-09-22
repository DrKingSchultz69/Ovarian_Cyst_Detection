/** T6 -- visualization primitives.
 *
 *  Inline SVG rather than a charting library. The project's dependency list is
 *  next + react + react-dom and nothing else; adding Recharts to draw eight
 *  bars would be the largest dependency in the tree. These are small enough to
 *  read in full, which also means the axis and scale decisions are visible
 *  rather than buried in a config object.
 *
 *  Shared conventions:
 *    - every chart states its own units in a caption, because a number
 *      without a unit is the commonest visualization failure in this domain
 *    - colour is never the only encoding: position or length always carries
 *      the value too, so the charts survive greyscale and colour-blindness
 *    - no truncated bar axes. A bar chart starting at a non-zero baseline
 *      misrepresents ratios, which in a measurement context is not a style
 *      preference.
 */

const AXIS = "rgba(255,255,255,0.18)";
const TEXT = "rgba(255,255,255,0.55)";
const TEXT_DIM = "rgba(255,255,255,0.35)";

export const SERIES = [
  "#38bdf8", // sky
  "#34d399", // emerald
  "#fbbf24", // amber
  "#f472b6", // pink
  "#a78bfa", // violet
  "#fb7185", // rose
] as const;

function niceTicks(max: number, count = 4): number[] {
  if (max <= 0) return [0];
  const raw = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const ticks: number[] = [];
  for (let v = 0; v <= max * 1.0001; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

const fmt = (v: number) =>
  Math.abs(v) >= 1000
    ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : Math.abs(v) >= 10
      ? v.toFixed(1)
      : v.toFixed(2);

// ---------------------------------------------------------------------------

export interface BarDatum {
  label: string;
  value: number;
  colour?: string;
  hint?: string;
}

/** Horizontal bars. Used wherever labels are words rather than numbers --
 *  vertical bars with rotated category labels are harder to read and this
 *  domain has long ones ("Global histogram equalisation"). */
export function BarChart({
  data,
  caption,
  unit,
  highlight,
  diverging = false,
}: {
  data: BarDatum[];
  caption: string;
  unit?: string;
  /** Label to draw in the accent colour, e.g. the best-performing method. */
  highlight?: string;
  /** Centre the axis on zero. For signed quantities like bias, where the sign
   *  is the point and a left-anchored bar would hide it. */
  diverging?: boolean;
}) {
  if (!data.length) return null;

  const rowHeight = 26;
  const labelWidth = 178;
  const chartWidth = 300;
  const height = data.length * rowHeight + 26;

  const values = data.map((d) => d.value);
  const max = Math.max(...values.map(Math.abs), 1e-9);
  const min = diverging ? -max : 0;
  const span = max - min || 1;
  const xOf = (v: number) => labelWidth + ((v - min) / span) * chartWidth;
  const zeroX = xOf(0);

  return (
    <figure className="w-full">
      <svg
        viewBox={`0 0 ${labelWidth + chartWidth + 62} ${height}`}
        className="w-full"
        role="img"
        aria-label={caption}
      >
        {niceTicks(max).map((t) => {
          const marks = diverging && t > 0 ? [-t, t] : [t];
          return marks.map((m) => (
            <g key={`${t}-${m}`}>
              <line
                x1={xOf(m)}
                x2={xOf(m)}
                y1={14}
                y2={height - 12}
                stroke={AXIS}
                strokeDasharray={m === 0 ? undefined : "2 3"}
              />
              <text x={xOf(m)} y={10} fill={TEXT_DIM} fontSize={9} textAnchor="middle">
                {fmt(m)}
              </text>
            </g>
          ));
        })}

        {data.map((d, i) => {
          const y = 18 + i * rowHeight;
          const x = xOf(Math.min(d.value, 0));
          const width = Math.abs(xOf(d.value) - zeroX);
          const colour =
            d.colour ?? (d.label === highlight ? SERIES[1] : SERIES[0]);
          return (
            <g key={d.label}>
              <text
                x={labelWidth - 8}
                y={y + 13}
                fill={d.label === highlight ? "#a7f3d0" : TEXT}
                fontSize={10.5}
                textAnchor="end"
              >
                {d.label.length > 30 ? `${d.label.slice(0, 29)}…` : d.label}
              </text>
              <rect
                x={diverging ? Math.min(zeroX, xOf(d.value)) : x}
                y={y + 3}
                width={Math.max(width, 1)}
                height={rowHeight - 12}
                fill={colour}
                opacity={0.82}
                rx={2}
              />
              <text
                x={(diverging ? Math.min(zeroX, xOf(d.value)) : x) + width + 6}
                y={y + 13}
                fill={TEXT}
                fontSize={10}
              >
                {fmt(d.value)}
              </text>
              {d.hint && <title>{d.hint}</title>}
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-1 text-xs text-white/40">
        {caption}
        {unit && <span className="ml-1 text-white/30">({unit})</span>}
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------

export interface LinePoint {
  x: number;
  y: number;
  label?: string;
}

export interface LineSeries {
  name: string;
  points: LinePoint[];
  colour?: string;
}

/** XY line chart with markers. Used for the rate-distortion curve, where both
 *  axes are continuous and the shape of the curve is the message. */
export function LineChart({
  series,
  caption,
  xLabel,
  yLabel,
  yMin,
  yMax,
  height = 220,
}: {
  series: LineSeries[];
  caption: string;
  xLabel: string;
  yLabel: string;
  yMin?: number;
  yMax?: number;
  height?: number;
}) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) return null;

  const width = 560;
  const pad = { left: 52, right: 14, top: 14, bottom: 34 };

  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = yMin ?? Math.min(...ys);
  const y1 = yMax ?? Math.max(...ys);

  const sx = (v: number) =>
    pad.left + ((v - x0) / (x1 - x0 || 1)) * (width - pad.left - pad.right);
  const sy = (v: number) =>
    height - pad.bottom - ((v - y0) / (y1 - y0 || 1)) * (height - pad.top - pad.bottom);

  const yTicks = 4;

  return (
    <figure className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={caption}>
        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const v = y0 + ((y1 - y0) * i) / yTicks;
          return (
            <g key={i}>
              <line
                x1={pad.left}
                x2={width - pad.right}
                y1={sy(v)}
                y2={sy(v)}
                stroke={AXIS}
                strokeDasharray={i === 0 ? undefined : "2 3"}
              />
              <text x={pad.left - 7} y={sy(v) + 3} fill={TEXT_DIM} fontSize={9.5} textAnchor="end">
                {fmt(v)}
              </text>
            </g>
          );
        })}

        {series.map((s, si) => {
          const colour = s.colour ?? SERIES[si % SERIES.length];
          const sorted = [...s.points].sort((a, b) => a.x - b.x);
          const path = sorted
            .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
            .join(" ");
          return (
            <g key={s.name}>
              <path d={path} fill="none" stroke={colour} strokeWidth={1.8} opacity={0.9} />
              {sorted.map((p, i) => (
                <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.8} fill={colour}>
                  <title>
                    {s.name} — {xLabel} {fmt(p.x)}, {yLabel} {fmt(p.y)}
                    {p.label ? ` (${p.label})` : ""}
                  </title>
                </circle>
              ))}
            </g>
          );
        })}

        <line
          x1={pad.left}
          x2={width - pad.right}
          y1={height - pad.bottom}
          y2={height - pad.bottom}
          stroke={AXIS}
        />
        <text x={(width + pad.left) / 2} y={height - 6} fill={TEXT_DIM} fontSize={10} textAnchor="middle">
          {xLabel}
        </text>
        <text
          x={12}
          y={height / 2}
          fill={TEXT_DIM}
          fontSize={10}
          textAnchor="middle"
          transform={`rotate(-90 12 ${height / 2})`}
        >
          {yLabel}
        </text>
      </svg>

      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">
        {series.length > 1 &&
          series.map((s, si) => (
            <span key={s.name} className="flex items-center gap-1.5 text-xs text-white/50">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: s.colour ?? SERIES[si % SERIES.length] }}
              />
              {s.name}
            </span>
          ))}
      </div>
      <figcaption className="mt-1 text-xs text-white/40">{caption}</figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------------------

/** Intensity histogram, computed in the browser from a rendered PNG.
 *
 *  Drawing it client-side avoids shipping 256 more numbers per panel from the
 *  API, and it means the histogram always matches the image actually on
 *  screen rather than one the server rendered separately.
 */
export function Histogram({
  counts,
  caption,
  colour = SERIES[0],
}: {
  counts: number[];
  caption: string;
  colour?: string;
}) {
  if (!counts.length) return null;
  const width = 256;
  const height = 74;

  // Scale to the tallest bin EXCLUDING pure black. A sector-scan ultrasound
  // frame is mostly black corners outside the field of view, so bin 0 holds
  // tens of thousands of pixels and every diagnostic intensity collapses to a
  // one-pixel smear against it. Bin 0 is still drawn -- it is real data -- but
  // it clips, and the caption says so rather than letting the reader assume
  // the axis is honest.
  const inField = counts.slice(1);
  const max = Math.max(...inField, 1);
  const zeroClips = counts[0] > max;

  return (
    <figure>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={caption}>
        <rect x={0} y={0} width={width} height={height} fill="rgba(255,255,255,0.02)" />
        {counts.map((c, i) => {
          const h = Math.min((c / max) * (height - 2), height - 2);
          return (
            <rect
              key={i}
              x={i}
              y={height - h}
              width={1}
              height={h}
              fill={i === 0 && zeroClips ? "rgba(255,255,255,0.22)" : colour}
              opacity={0.85}
            />
          );
        })}
      </svg>
      <figcaption className="mt-1 text-[11px] text-white/35">
        {caption}
        {zeroClips && <span className="text-white/25"> · bin 0 clipped (out-of-field black)</span>}
      </figcaption>
    </figure>
  );
}

/** 256-bin luminance histogram of a base64 PNG. Returns [] on any failure, so
 *  callers render the chart conditionally rather than handling an error.
 *
 *  Uses load/error events plus a timeout rather than `img.decode()`. decode()
 *  is the tidier API, but its promise can simply never settle when the tab is
 *  backgrounded -- the browser defers the decode and nothing rejects. That
 *  leaves the caller awaiting forever and the histogram silently absent, which
 *  is exactly the failure this function is least able to report.
 */
async function loadImage(src: string, timeoutMs = 8000): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const image = new Image();
    const done = (value: HTMLImageElement | null) => {
      clearTimeout(timer);
      image.onload = image.onerror = null;
      resolve(value);
    };
    const timer = setTimeout(() => done(null), timeoutMs);
    image.onload = () => done(image);
    image.onerror = () => done(null);
    image.src = src;
  });
}

export async function histogramOf(pngB64: string): Promise<number[]> {
  const image = await loadImage(`data:image/png;base64,${pngB64}`);
  if (!image || !image.width || !image.height) return [];

  const canvas = document.createElement("canvas");
  // Downsample: a 640x480 frame is 300k pixels and the shape of the histogram
  // is identical from a quarter of them, at a quarter of the cost.
  const scale = Math.min(1, 320 / Math.max(image.width, image.height));
  canvas.width = Math.max(1, Math.round(image.width * scale));
  canvas.height = Math.max(1, Math.round(image.height * scale));

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return [];

  let data: Uint8ClampedArray;
  try {
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    ({ data } = context.getImageData(0, 0, canvas.width, canvas.height));
  } catch {
    // getImageData throws on a tainted canvas. Data URLs are same-origin so
    // this should not fire, but a silent empty histogram beats a thrown error
    // inside an effect.
    return [];
  }

  const counts = new Array<number>(256).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    // Rec. 601 luma. The panels are greyscale-derived, but the PNGs are RGB.
    const y = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) | 0;
    counts[Math.min(255, Math.max(0, y))] += 1;
  }
  return counts;
}

// ---------------------------------------------------------------------------

/** Proportion bar for a single percentage. Used for missingness per column,
 *  where a full chart would be heavier than the information warrants. */
export function ProportionBar({
  pct,
  label,
  right,
  tone = "sky",
}: {
  pct: number;
  label: string;
  right?: string;
  tone?: "sky" | "amber" | "rose" | "emerald";
}) {
  const colour = { sky: SERIES[0], amber: SERIES[2], rose: SERIES[5], emerald: SERIES[1] }[tone];
  return (
    <div className="flex items-center gap-3">
      <span className="w-44 shrink-0 truncate text-xs text-white/60">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(100, Math.max(0, pct))}%`, background: colour }}
        />
      </div>
      <span className="w-20 shrink-0 text-right text-xs tabular-nums text-white/50">
        {right ?? `${pct.toFixed(1)}%`}
      </span>
    </div>
  );
}
