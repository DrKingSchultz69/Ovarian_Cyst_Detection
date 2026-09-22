"""Hugging Face Space — pure Gradio.

A Gradio Space runs `python app.py` and expects demo.launch() to own the
server. Mounting a custom FastAPI app instead gets the process killed at
startup, and the Docker SDK that would allow it is a paid feature -- so the
JSON API lives elsewhere (see modal_app.py) and this file is UI only.

All image logic is imported from inference.py unchanged, so this UI and the
API cannot drift apart.
"""

from __future__ import annotations

import base64
import io
import os

import gradio as gr
import numpy as np
from PIL import Image

# Ultralytics writes settings on import; /home/user/.config is read-only here.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import inference  # noqa: E402  (must follow the env vars above)

DISCLAIMER = (
    "### OvaScan v0.1 — research prototype. "
    "**Not a medical device. Not for clinical use.**"
)

COLUMNS = ["Lesion", "Confidence", "Max diameter (px)", "Area (px²)",
           "Circularity", "Solidity", "Mean echo (0-255)", "Echo SD"]

_HEAD = "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n"
EMPTY_TABLE = _HEAD + "| " + " | ".join(["—"] * len(COLUMNS)) + " |"


def _table(rows: list[list[str]]) -> str:
    """Markdown, not gr.Dataframe: the Dataframe schema carries a `metadata`
    dict that gradio 5.9.1's API-schema generator crashes on, which 500s the
    page itself (get_api_info -> 'bool is not iterable')."""
    if not rows:
        return EMPTY_TABLE
    return _HEAD + "\n".join("| " + " | ".join(r) + " |" for r in rows)


def _img(b64: str | None) -> Image.Image | None:
    return Image.open(io.BytesIO(base64.b64decode(b64))) if b64 else None


def analyse(image: np.ndarray | None, conf: float):
    if image is None:
        return None, EMPTY_TABLE, "Upload an image to begin.", None, None, None

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    result = inference.analyse_bytes(buf.getvalue(), conf=conf, include_debug=True)

    if not result["ok"]:
        return None, EMPTY_TABLE, f"⚠️ {result['error']}", None, None, None

    rows = [
        [f"#{l['id']}", f"{l['conf']:.3f}", f"{l['max_diameter_px']:.1f}",
         f"{l['area_px']:,}", f"{l['circularity']:.3f}", f"{l['solidity']:.3f}",
         f"{l['echo_mean']:.1f}", f"{l['echo_sd']:.1f}"]
        for l in result["lesions"]
    ]
    status = (
        f"**No lesion detected** at conf ≥ {conf:.2f} · {result['latency_ms']} ms"
        if result["count"] == 0
        else f"**{result['count']} lesion(s)** · conf ≥ {conf:.2f} · {result['latency_ms']} ms"
    )
    dbg = result.get("debug", {})
    return (
        _img(result["annotated_png"]), _table(rows), status,
        _img(dbg.get("raw_png")), _img(dbg.get("burnin_mask_png")),
        _img(dbg.get("inpainted_png")),
    )


with gr.Blocks(title="OvaScan v0.1") as demo:
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Ultrasound image", type="numpy", height=340)
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                             label="Confidence threshold")
            run = gr.Button("Analyse", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Segmentation + detection", height=340)
            status = gr.Markdown("Ready.")

    gr.Markdown("#### Measurements — **all values in image pixels**")
    table = gr.Markdown(EMPTY_TABLE)
    gr.Markdown(
        "_mm conversion needs the scanner scale bar or DICOM pixel spacing — "
        "out of scope for this prototype._"
    )

    with gr.Accordion("Show preprocessing (burn-in removal)", open=False):
        with gr.Row():
            pp_raw = gr.Image(label="1 · Raw", height=240)
            pp_mask = gr.Image(label="2 · Burn-in mask", height=240)
            pp_inp = gr.Image(label="3 · Inpainted", height=240)
        gr.Markdown(
            "Sonographer calipers sit inside the lesion. Left in place, the model "
            "learns *lesion = wherever the calipers are* and fails on unmarked "
            "images, so they are detected by saturation and brightness, then "
            "Telea-inpainted before inference."
        )

    outs = [img_out, table, status, pp_raw, pp_mask, pp_inp]
    img_in.change(analyse, [img_in, conf], outs)
    conf.release(analyse, [img_in, conf], outs)
    run.click(analyse, [img_in, conf], outs)

if __name__ == "__main__":
    # show_api=False: gradio 5.9.1's schema generator crashes on this Blocks
    #   (TypeError: argument of type 'bool' is not iterable in get_api_info).
    # ssr_mode=False: the SSR node server calls /info at startup and dies with it.
    demo.launch(show_api=False, ssr_mode=False)
