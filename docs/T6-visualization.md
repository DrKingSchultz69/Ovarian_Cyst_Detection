# T6 — Biomedical image visualization

> Syllabus: *"T6: Biomedical Image - Visualization"*

**Code:** [`components/Charts.tsx`](../frontend/src/components/Charts.tsx),
[`components/Ui.tsx`](../frontend/src/components/Ui.tsx),
[`components/Preprocessing.tsx`](../frontend/src/components/Preprocessing.tsx),
`_annotate()` in [`inference.py`](../backend/inference.py) · **Pages:** all four

---

## The principle

Visualization in a clinical tool is not decoration and it is not persuasion. Its job is
to let a viewer **check the result** — and, more importantly, to notice when it is
wrong. Every choice below follows from that.

Concretely, this project never shows a processed image without the number that says
whether the processing helped. An enhancement demo without metrics is a filter gallery;
it invites the viewer to prefer whichever image looks nicest, which on ultrasound is
frequently the one that destroyed the most information.

---

## 1. Overlay — segmentation

`_annotate()` in `inference.py`.

| Layer | Encoding | Why |
|---|---|---|
| Mask fill | amber, **35% alpha** | the underlying texture must stay visible — an opaque mask hides the thing being judged |
| Boundary | bright green, 2 px, antialiased | precise extent; the fill alone is too soft to localise an edge |
| Box + label | green with class and confidence | detection-level result, separate from the mask |

Fill *and* outline is redundant on purpose. The fill makes the region obvious at a
glance; the outline makes the boundary checkable. Either alone fails one of the two
jobs.

`count: 0` renders as a labelled negative result — "No lesion detected" drawn on the
frame — not an error state. A negative is a finding.

## 2. Pipeline panels — restoration and enhancement

`Preprocessing.tsx` shows the three burn-in stages: raw → mask → inpainted. The
enhancement strip on `/imaging` extends the same idea to nine panels.

Showing intermediates is what makes a pipeline auditable rather than a black box. The
burn-in mask panel in particular lets a viewer see *immediately* if the mask is eating
anatomy — which the final image alone would hide, because inpainted anatomy looks
plausible.

Each enhancement panel carries its intensity histogram and four metrics beneath the
image.

## 3. Histograms

`Histogram` + `histogramOf` in `Charts.tsx`. 256-bin luminance, computed **client-side**
from the rendered PNG.

Client-side because it avoids shipping 256 more numbers per panel, and because the
histogram then always matches the image actually on screen rather than one the server
rendered separately.

Two implementation notes that are really visualization decisions:

**Bin 0 is clipped, and the caption says so.** A sector-scan frame is mostly black
corners outside the field of view, so bin 0 holds tens of thousands of pixels. Scaling
to it collapses every diagnostic intensity into a one-pixel smear — the chart is
technically honest and practically useless. The axis is scaled to the tallest
*in-field* bin, bin 0 is drawn in a dimmer colour and clips, and the caption reads
"bin 0 clipped (out-of-field black)". Silently dropping bin 0 would have been the easy
fix and the dishonest one.

**Downsampled to ≤320 px before counting.** The histogram shape is identical from a
quarter of the pixels at a quarter of the cost.

## 4. Charts

Inline SVG, no charting library. The project's dependencies are `next`, `react`,
`react-dom` and nothing else; Recharts would be the largest thing in the tree, added to
draw eight bars. The charts are small enough to read in full, which also means the axis
and scale decisions are visible rather than buried in a config object.

### Rules the charts follow

**No truncated bar axes.** A bar chart starting at a non-zero baseline misrepresents
ratios. In a measurement context that is not a style preference.

**Colour is never the only encoding.** Position or length always carries the value too,
so the charts survive greyscale printing and colour vision deficiency. Colour marks
emphasis — best performer, above-threshold correlation — and is redundant with the
number printed beside every bar.

**Signed quantities get a diverging axis centred on zero.** Imputation bias is the case
that matters: the *sign* is the finding — whether a method runs high or low — and a
left-anchored bar hides it.

**Every chart states its units.** A number without a unit is the commonest
visualization failure in this domain, and this project's measurements are in pixels,
not millimetres, which is exactly the kind of thing a viewer assumes wrongly.

**Long category labels get horizontal bars.** "Global histogram equalisation" as a
rotated axis label is unreadable.

### The charts, and what each is for

| Chart | Where | Job |
|---|---|---|
| `BarChart` | enhancement metrics, flags by rule, linkage outcomes | compare categories |
| `BarChart` (diverging) | imputation bias, missingness correlations | show sign |
| `LineChart` | rate-distortion curves | show the shape of a trade-off |
| `Histogram` | every enhancement panel | intensity distribution |
| `ProportionBar` | missingness per column | a percentage where a full chart is overkill |

## 5. Registration views

Four, because each fails to show something the others catch:

1. **Inlier matches** — which correspondences RANSAC accepted. Catches "it matched the
   wrong structure entirely".
2. **Warped follow-up** — the result on its own.
3. **Checkerboard** — the standard registration QA view. Misalignment makes structure
   break at tile boundaries, which the eye catches instantly.
4. **Difference map** — `INFERNO`-mapped absolute difference. Bright means poorly
   aligned.

The checkerboard is the one that earns its place: a difference image of two noisy
ultrasound frames is bright almost everywhere because the speckle differs, so it looks
alarming even for a perfect alignment. The checkerboard shows *structural* continuity
instead, which is what actually matters.

## 6. Tables

`DataTable` in `Ui.tsx`. Tabular figures throughout, so columns of numbers align and a
changing value does not make the row jitter. Wide tables scroll inside their own box —
the page never scrolls sideways.

Tables are used, not avoided, where the reader needs exact values. The rate-distortion
curve shows the shape; the table beneath it gives the numbers to quote. Both, not
either.

## 7. Colour

A dark surround throughout, matching how ultrasound is actually read — a bright
surround makes the eye adapt while it is judging low-echo regions.

Six-colour categorical series, and `COLORMAP_INFERNO` for the difference map:
perceptually uniform and monotonic in lightness, so it does not invent banding at
values that happen to fall on a hue boundary. `COLORMAP_HOT` on the burn-in mask is the
pre-existing choice and stays, since it is a binary mask where perceptual uniformity is
not in play.

---

## Visualizations deliberately not built

- **3D / volume rendering.** The MMOTU data is 2D B-mode. There is no volume to render,
  and faking one would misrepresent the data.
- **Interactive ROI drawing.** The phantom supplies a known lesion box; an uploaded
  image gets no ROI, and the page says so rather than guessing one.
- **Side-by-side pixel-peeping zoom** for compression artifacts. The three quality
  examples are shown at full width instead; a magnifier would be better and is not
  built.

---

Previous: [T5](T5-sift-ransac-cnn.md) · Index: [README](README.md)
