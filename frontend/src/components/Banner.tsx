import { usingMock } from "@/lib/api";

/** Fixed, always visible. This is a research prototype and the UI must say so
 *  on every screen, not only in an about page. */
export function Banner() {
  return (
    <header className="sticky top-0 z-20 border-b border-amber-500/30 bg-amber-500/10 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2.5 text-sm">
        <span className="font-semibold tracking-tight text-amber-200">OvaScan v0.2</span>
        <span className="text-amber-100/70">
          Research prototype. <strong className="font-semibold">Not a medical device. Not for clinical use.</strong>
        </span>
        {usingMock && (
          <span className="ml-auto rounded-full border border-sky-400/40 bg-sky-400/10 px-2.5 py-0.5 text-xs font-medium text-sky-200">
            mock data — no backend connected
          </span>
        )}
      </div>
    </header>
  );
}
