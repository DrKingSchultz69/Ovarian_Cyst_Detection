"use client";

import { useCallback, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  fileName: string | null;
  disabled: boolean;
}

const ACCEPT = "image/jpeg,image/png,image/bmp,image/tiff,image/webp";

export function Uploader({ onFile, fileName, disabled }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = useCallback(
    (files: FileList | null) => {
      const f = files?.[0];
      if (f) onFile(f);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) take(e.dataTransfer.files);
      }}
      className={`rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
        over ? "border-sky-400 bg-sky-400/5" : "border-white/15 bg-white/[0.02]"
      } ${disabled ? "opacity-50" : ""}`}
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => take(e.target.files)}
      />
      <p className="text-sm text-white/70">Drop an ultrasound image here</p>
      <button
        type="button"
        disabled={disabled}
        onClick={() => input.current?.click()}
        className="mt-3 rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-white/10"
      >
        Choose file
      </button>
      <p className="mt-3 truncate text-xs text-white/40">
        {fileName ?? "JPG, PNG, BMP, TIF or WebP · max 10 MB"}
      </p>
    </div>
  );
}
