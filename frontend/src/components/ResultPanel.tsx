import type { PredictResponse } from "@/lib/types";

interface Props {
  result: PredictResponse | null;
  previewUrl: string | null;
  loading: boolean;
}

/** Four states, all handled here: idle, loading, negative (count 0), error. */
export function ResultPanel({ result, previewUrl, loading }: Props) {
  const frame =
    "flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border border-white/10 bg-black";

  if (loading) {
    return (
      <div className={frame}>
        <div className="flex flex-col items-center gap-3 text-sm text-white/50">
          <span className="h-6 w-6 animate-spin rounded-full border-2 border-white/20 border-t-sky-400" />
          Running inference…
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className={frame}>
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={previewUrl} alt="Selected image" className="max-h-full max-w-full opacity-40" />
        ) : (
          <p className="px-6 text-center text-sm text-white/35">
            Upload an image to see the segmentation overlay.
          </p>
        )}
      </div>
    );
  }

  if (!result.ok) {
    return (
      <div className="flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-xl border border-red-500/30 bg-red-500/[0.06] p-6">
        <div className="text-center">
          <p className="text-sm font-medium text-red-300">Could not process this image</p>
          <p className="mt-2 break-words text-xs leading-relaxed text-red-200/70">{result.error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className={frame}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${result.annotated_png}`}
          alt={`${result.count} lesion(s) detected`}
          className="max-h-full max-w-full"
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-white/50">
        {result.count === 0 ? (
          <span className="font-medium text-amber-300">No lesion detected</span>
        ) : (
          <span className="font-medium text-emerald-300">
            {result.count} lesion{result.count > 1 ? "s" : ""} detected
          </span>
        )}
        <span>conf ≥ {result.conf.toFixed(2)}</span>
        <span>{result.latency_ms} ms</span>
        <span>
          {result.image_size.width} × {result.image_size.height} px
        </span>
      </div>
    </div>
  );
}
