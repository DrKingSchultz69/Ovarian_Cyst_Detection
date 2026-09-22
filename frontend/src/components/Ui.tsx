/** Shared layout and table primitives.
 *
 *  Extracted so the four topic pages stay readable: without these each page
 *  would repeat the same forty lines of Tailwind for a panel and a table.
 */

import type { ReactNode } from "react";

export function PageHeader({
  title,
  topics,
  children,
}: {
  title: string;
  topics: string[];
  children?: ReactNode;
}) {
  return (
    <header className="mb-6">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="text-lg font-semibold tracking-tight text-white/90">{title}</h1>
        {topics.map((t) => (
          <span
            key={t}
            className="rounded-full border border-sky-400/30 bg-sky-400/10 px-2 py-0.5 text-[11px] font-medium text-sky-200"
          >
            {t}
          </span>
        ))}
      </div>
      {children && (
        <div className="mt-2 max-w-3xl text-sm leading-relaxed text-white/50">{children}</div>
      )}
    </header>
  );
}

export function Section({
  id,
  title,
  topic,
  subtitle,
  children,
}: {
  id?: string;
  title: string;
  topic?: string;
  subtitle?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="mt-8 scroll-mt-20">
      <div className="flex flex-wrap items-baseline gap-2">
        <h2 className="text-sm font-semibold tracking-tight text-white/90">{title}</h2>
        {topic && (
          <span className="rounded bg-white/[0.07] px-1.5 py-0.5 text-[10px] font-medium text-white/45">
            {topic}
          </span>
        )}
      </div>
      {subtitle && (
        <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-white/45">{subtitle}</p>
      )}
      <div className="mt-3">{children}</div>
    </section>
  );
}

export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-white/10 bg-white/[0.02] p-4 ${className}`}>
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  const colour = {
    neutral: "text-white/90",
    good: "text-emerald-300",
    warn: "text-amber-300",
    bad: "text-rose-300",
  }[tone];
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wide text-white/35">{label}</div>
      <div className={`mt-0.5 text-lg font-semibold tabular-nums ${colour}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] leading-snug text-white/35">{hint}</div>}
    </div>
  );
}

export interface Column<T> {
  key: string;
  label: string;
  unit?: string;
  align?: "left" | "right";
  render?: (row: T) => ReactNode;
}

/** Scrolls inside its own box so the page never scrolls sideways -- the same
 *  rule MeasurementsTable already follows.
 *
 *  T is deliberately unconstrained. `T extends Record<string, unknown>` reads
 *  like the right bound but rejects every plain interface, because an
 *  interface has no implicit index signature -- so each caller would have to
 *  add one to a type that otherwise wants to stay closed. The fallback cell
 *  lookup casts instead, and it is only reached for columns without a
 *  `render`, where the value is a primitive by construction. */
export function DataTable<T>({
  columns,
  rows,
  empty = "Nothing to show yet.",
  minWidth = 640,
  rowKey,
  highlight,
}: {
  columns: Column<T>[];
  rows: T[];
  empty?: string;
  minWidth?: number;
  rowKey?: (row: T, i: number) => string;
  highlight?: (row: T) => boolean;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full border-collapse text-sm" style={{ minWidth }}>
        <thead>
          <tr className="bg-white/[0.04] text-left">
            {columns.map((c) => (
              <th
                key={c.key}
                className={`px-3 py-2 font-medium text-white/70 ${
                  c.align === "right" ? "text-right" : ""
                }`}
              >
                {c.label}
                {c.unit && (
                  <span className="ml-1 text-xs font-normal text-white/35">({c.unit})</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-3 py-6 text-center text-white/40">
                {empty}
              </td>
            </tr>
          ) : (
            rows.map((row, i) => (
              <tr
                key={rowKey ? rowKey(row, i) : i}
                className={`border-t border-white/[0.07] hover:bg-white/[0.03] ${
                  highlight?.(row) ? "bg-emerald-400/[0.06]" : ""
                }`}
              >
                {columns.map((c) => (
                  <td
                    key={c.key}
                    className={`px-3 py-2 tabular-nums text-white/85 ${
                      c.align === "right" ? "text-right" : ""
                    }`}
                  >
                    {c.render
                      ? c.render(row)
                      : String((row as Record<string, unknown>)[c.key] ?? "—")}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export function Callout({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "good";
  title?: string;
  children: ReactNode;
}) {
  const style = {
    info: "border-sky-400/25 bg-sky-400/[0.06] text-sky-100/75",
    warn: "border-amber-400/25 bg-amber-400/[0.06] text-amber-100/75",
    good: "border-emerald-400/25 bg-emerald-400/[0.06] text-emerald-100/75",
  }[tone];
  return (
    <div className={`rounded-lg border p-3 text-xs leading-relaxed ${style}`}>
      {title && <div className="mb-1 font-semibold">{title}</div>}
      {children}
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 py-8 text-sm text-white/50">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-sky-400" />
      {label}
    </div>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/[0.06] p-4">
      <p className="text-sm font-medium text-red-300">Could not load this</p>
      <p className="mt-1.5 break-words text-xs leading-relaxed text-red-200/70">{error}</p>
    </div>
  );
}
