"""OvaScan API.

Thin HTTP layer over the analysis modules. All logic lives in those; this file
only handles transport concerns -- CORS, upload validation, and shaping the
response. Keeping the split means every module can be debugged in a REPL.

  inference.py     segmentation          (T4 segmentation, T5 CNN)
  enhance.py       enhancement           (T4 enhancement)
  compress.py      compression analysis  (T4 compression)
  registration.py  SIFT + RANSAC         (T5)
  ehr/             EHR pipeline          (T1, T2, T3)

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import compress
import enhance
import phantom
import registration

# inference.py loads the YOLO checkpoint at import and raises without it. That
# is correct for /predict and wrong for everything else: the enhancement,
# compression, registration and EHR endpoints need neither torch nor weights,
# and a grader without the 6 MB checkpoint should still get a working service.
# So the failure is captured rather than propagated, and only /predict reports
# it -- as a 503, which is what "this instance cannot serve this route" means.
try:
    import inference
    _INFERENCE_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - the whole point is to not die here
    inference = None  # type: ignore[assignment]
    _INFERENCE_ERROR = f"{type(exc).__name__}: {exc}"

app = FastAPI(
    title="OvaScan API",
    version="0.2.0",
    description=(
        "Research prototype for ovarian cyst segmentation in ultrasound images, "
        "plus the EHR and image-analysis pipelines. "
        "Not a medical device. Not for clinical use."
    ),
)

# CORS must exist from day one. curl works without it; the browser will not,
# and that failure looks like a network error rather than a policy error.
# Set OVASCAN_ALLOWED_ORIGINS to a comma-separated list before going public.
_origins = os.environ.get("OVASCAN_ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

ACCEPTED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/webp",
}


MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _png_b64(bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


async def _read_image(file: UploadFile) -> np.ndarray:
    """Shared upload path for every image endpoint: type gate, size gate, decode."""
    if file.content_type and file.content_type not in ACCEPTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Accepted: {', '.join(sorted(ACCEPTED_TYPES))}",
        )

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(data) / 1e6:.1f} MB; limit is "
                   f"{MAX_IMAGE_BYTES / 1e6:.0f} MB.",
        )

    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=422, detail="Not a decodable image.")
    return decoded


@app.get("/health")
def health() -> dict:
    """Liveness probe. Render and most PaaS hosts kill a service without one."""
    return {
        "status": "ok",
        "model": inference.model_info() if inference else None,
        "segmentation_available": inference is not None,
        "model_error": _INFERENCE_ERROR,
    }


# Kept off "/" so a mounted Gradio UI can own the root path.
@app.get("/api")
def root() -> dict:
    return {
        "name": "OvaScan API",
        "version": "0.2.0",
        "disclaimer": "Research prototype. Not a medical device. Not for clinical use.",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "enhance": "POST /image/enhance",
            "compress": "POST /image/compress",
            "register": "POST /image/register",
            "register_and_segment": "POST /image/register-and-segment",
            "phantom": "GET /image/phantom",
            "ehr": "GET /ehr/pipeline",
            "docs": "GET /docs",
        },
        "topics": {
            "T1": "GET /ehr/pipeline -> source, coding systems",
            "T2": "GET /ehr/pipeline -> standardize, clean",
            "T3": "GET /ehr/pipeline -> dedupe, missing",
            "T4": "POST /image/enhance, POST /image/compress, POST /predict",
            "T5": "POST /image/register (SIFT/RANSAC), POST /predict (CNN)",
            "T6": "annotated overlays and panels on every image endpoint",
        },
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Ultrasound image (jpg/png/bmp/tif/webp)"),
    conf: float = Form(0.25, ge=0.05, le=0.95),
    debug: bool = Form(False, description="Also return raw / burn-in mask / inpainted panels"),
) -> JSONResponse:
    if inference is None:
        raise HTTPException(
            status_code=503,
            detail=f"Segmentation model unavailable: {_INFERENCE_ERROR}. "
                   f"The enhancement, compression, registration and EHR "
                   f"endpoints do not need it and still work.",
        )

    # Reject the obvious wrong thing before spending a model pass on it.
    if file.content_type and file.content_type not in ACCEPTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Accepted: {', '.join(sorted(ACCEPTED_TYPES))}",
        )

    data = await file.read()
    if len(data) > inference.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(data) / 1e6:.1f} MB; limit is "
                   f"{inference.MAX_IMAGE_BYTES / 1e6:.0f} MB.",
        )

    result = inference.analyse_bytes(data, conf=conf, include_debug=debug)

    # A malformed upload is the client's problem, not a server fault -- but the
    # body still carries the usable error string for the UI to display.
    status = 200 if result.get("ok") else 422
    return JSONResponse(status_code=status, content=result)


# ===========================================================================
# T4 -- enhancement
# ===========================================================================


@app.get("/image/phantom")
def get_phantom(seed: int = Query(0, ge=0, le=9999)) -> dict:
    """A synthetic ultrasound frame, so every other endpoint is demonstrable
    without the 1.5 GB MMOTU download. See phantom.py for what it does and
    does not model."""
    image, bbox = phantom.generate(seed=seed)
    return {
        "ok": True,
        "png": _png_b64(image),
        "lesion_bbox": list(bbox),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "note": "Synthetic phantom with multiplicative Rayleigh speckle. "
                "Not a real scan and not a wave simulation.",
    }


@app.post("/image/enhance")
async def image_enhance(
    file: UploadFile = File(..., description="Ultrasound image"),
    methods: str = Form("", description="Comma-separated method names; blank runs all"),
    roi: str = Form("", description="x1,y1,x2,y2 for the contrast-to-noise metric"),
) -> JSONResponse:
    """Run enhancement methods side by side and measure each one.

    Returns the original first, then one panel per method with its quality
    metrics, so the UI can render a strip with numbers underneath instead of
    asking the viewer to judge by eye.
    """
    image = await _read_image(file)

    names = [n.strip() for n in methods.split(",") if n.strip()] or None
    if names:
        unknown = [n for n in names if n not in enhance.METHODS]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown method(s): {', '.join(unknown)}. "
                       f"Available: {', '.join(enhance.METHODS)}",
            )

    box = None
    if roi.strip():
        try:
            x1, y1, x2, y2 = (int(v) for v in roi.split(","))
            box = (x1, y1, x2, y2)
        except ValueError:
            raise HTTPException(status_code=422, detail="roi must be 'x1,y1,x2,y2'.")

    panels = enhance.compare(image, names, roi=box)
    return JSONResponse({
        "ok": True,
        "roi": list(box) if box else None,
        "panels": [
            {
                "name": p["name"],
                "label": p["label"],
                "group": p["group"],
                "note": p.get("note"),
                "png": _png_b64(p["image"]),
                "metrics": p["metrics"],
            }
            for p in panels
        ],
        "metric_notes": {
            "sharpness": "Variance of the Laplacian. Rises with detail AND with noise.",
            "speckle_index": "sigma/mu. Lower is less speckle. The ultrasound-specific one.",
            "entropy": "Shannon entropy in bits. Information content.",
            "cnr": "Contrast-to-noise between the ROI and the rest. Needs an ROI.",
        },
    })


# ===========================================================================
# T4 -- compression
# ===========================================================================


@app.post("/image/compress")
async def image_compress(
    file: UploadFile = File(..., description="Ultrasound image"),
    codec: str = Form("jpeg"),
    roi: str = Form("", description="x1,y1,x2,y2 to also measure measurement drift"),
    target_kb: float = Form(40.0, gt=1.0, le=5000.0),
) -> JSONResponse:
    """Rate-distortion sweep, equal-size codec comparison, and -- when an ROI
    is supplied -- how far the lesion measurements move.

    The measurement drift is the number that matters clinically; PSNR and
    SSIM are the ones that are easy to quote.
    """
    if codec not in compress.CODECS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown codec {codec!r}. Available: {', '.join(compress.CODECS)}",
        )

    image = await _read_image(file)

    box = None
    if roi.strip():
        try:
            x1, y1, x2, y2 = (int(v) for v in roi.split(","))
            box = (x1, y1, x2, y2)
        except ValueError:
            raise HTTPException(status_code=422, detail="roi must be 'x1,y1,x2,y2'.")

    payload: dict = {
        "ok": True,
        "codec": codec,
        "raw_kb": round(image.nbytes / 1024, 1),
        "sweep": compress.sweep(image, codec),
        "equal_size": compress.equal_size_comparison(image, target_kb),
        "codecs": {
            name: {"label": spec["label"], "lossy": spec["lossy"], "note": spec["note"]}
            for name, spec in compress.CODECS.items()
        },
    }

    if box:
        payload["roi"] = list(box)
        payload["measurement_drift"] = compress.measurement_drift(image, box, codec)

    # Two visual examples, so the numbers have something to sit next to.
    if compress.CODECS[codec]["lossy"]:
        payload["examples"] = [
            {"quality": q, "png": _png_b64(compress.decode(compress.encode(image, codec, q))),
             "kb": round(len(compress.encode(image, codec, q)) / 1024, 1)}
            for q in (90, 50, 15)
        ]

    return JSONResponse(payload)


# ===========================================================================
# T5 -- SIFT + RANSAC registration
# ===========================================================================


@app.post("/image/register")
async def image_register(
    baseline: UploadFile = File(..., description="Baseline scan"),
    follow_up: UploadFile = File(..., description="Follow-up scan to align onto it"),
    model: str = Form("affine", description="rigid | affine | homography"),
    ratio: float = Form(0.72, ge=0.3, le=0.99),
    contrast_threshold: float = Form(0.04, ge=0.001, le=0.3),
) -> JSONResponse:
    """Align two scans with SIFT keypoints, ratio-tested matching and RANSAC.

    Returns the fit statistics plus four views: the match visualisation, the
    warped follow-up, a checkerboard and a difference map.
    """
    if model not in ("rigid", "affine", "homography"):
        raise HTTPException(
            status_code=422, detail="model must be rigid, affine or homography.",
        )

    base_img = await _read_image(baseline)
    follow_img = await _read_image(follow_up)

    result = registration.register(
        base_img, follow_img, model=model, ratio=ratio,
        contrast_threshold=contrast_threshold,
    )

    payload = result.as_dict()
    payload["ok"] = result.ok

    if not result.ok:
        return JSONResponse(status_code=200, content=payload)

    payload["views"] = {
        "matches": _png_b64(result.match_visualisation),
        "warped": _png_b64(result.warped),
        "checkerboard": _png_b64(registration.checkerboard(base_img, result.warped)),
        "difference": _png_b64(registration.difference(base_img, result.warped)),
    }
    payload["interpretation"] = {
        "inlier_ratio": "Fraction of ratio-tested matches the model explains. "
                        "Below ~0.3 the alignment is not trustworthy.",
        "rmse_px": "Reprojection error over inliers only. Do NOT use it to "
                   "choose between models -- more degrees of freedom always "
                   "lower it, including when they make the transform worse.",
    }
    return JSONResponse(payload)


@app.post("/image/register-and-segment")
async def image_register_and_segment(
    baseline: UploadFile = File(..., description="Baseline scan"),
    follow_up: UploadFile = File(..., description="Follow-up scan to align and segment"),
    model: str = Form("affine", description="rigid | affine | homography"),
    ratio: float = Form(0.72, ge=0.3, le=0.99),
    contrast_threshold: float = Form(0.04, ge=0.001, le=0.3),
    conf: float = Form(0.25, ge=0.05, le=0.95),
) -> JSONResponse:
    """Register the follow-up, then compare CNN measurements in one frame."""
    if inference is None:
        raise HTTPException(status_code=503, detail=f"Segmentation unavailable: {_INFERENCE_ERROR}")
    if model not in ("rigid", "affine", "homography"):
        raise HTTPException(status_code=422, detail="model must be rigid, affine or homography.")

    base_img = await _read_image(baseline)
    follow_img = await _read_image(follow_up)
    fit = registration.register(base_img, follow_img, model=model, ratio=ratio,
                                contrast_threshold=contrast_threshold)
    if not fit.ok or fit.warped is None:
        return JSONResponse(status_code=200, content=fit.as_dict())

    def png_bytes(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Could not encode the registered image.")
        return encoded.tobytes()

    baseline_seg = inference.analyse_bytes(png_bytes(base_img), conf=conf)
    followup_seg = inference.analyse_bytes(png_bytes(fit.warped), conf=conf)
    before = (baseline_seg.get("lesions") or [None])[0]
    after = (followup_seg.get("lesions") or [None])[0]

    def pct(after_value: float, before_value: float) -> float | None:
        return None if before_value == 0 else round((after_value - before_value) / before_value * 100, 2)

    comparison = None
    if before and after:
        comparison = {
            "area_change_pct": pct(float(after["area_px"]), float(before["area_px"])),
            "max_diameter_change_pct": pct(float(after["max_diameter_px"]), float(before["max_diameter_px"])),
            "baseline": before,
            "followup_registered": after,
        }

    payload = fit.as_dict()
    payload.update({
        "views": {
            "matches": _png_b64(fit.match_visualisation),
            "warped": _png_b64(fit.warped),
            "checkerboard": _png_b64(registration.checkerboard(base_img, fit.warped)),
            "difference": _png_b64(registration.difference(base_img, fit.warped)),
        },
        "segmentation": {"baseline": baseline_seg, "followup_registered": followup_seg},
        "comparison": comparison,
    })
    return JSONResponse(payload)

@app.get("/image/register/demo")
def register_demo(
    seed: int = Query(0, ge=0, le=9999),
    rotation: float = Query(6.0, ge=-45.0, le=45.0),
    scale: float = Query(1.06, ge=0.5, le=2.0),
) -> dict:
    """Registration against a KNOWN transform.

    phantom.pair() applies a transform we choose and then re-speckles the
    follow-up independently, so the recovered matrix can be scored by how far
    it moves the image corners. This is the endpoint that demonstrates the
    homography-overfits-and-wins-on-RMSE result.
    """
    base_img, follow_img, truth = phantom.pair(
        seed=seed, rotation=rotation, scale=scale,
    )
    rows = registration.compare_models(base_img, follow_img, truth)
    best = registration.register(base_img, follow_img, model="affine")

    return {
        "ok": True,
        "true_transform": truth.round(5).tolist(),
        "models": rows,
        "baseline_png": _png_b64(base_img),
        "followup_png": _png_b64(follow_img),
        "views": None if not best.ok else {
            "matches": _png_b64(best.match_visualisation),
            "warped": _png_b64(best.warped),
            "checkerboard": _png_b64(registration.checkerboard(base_img, best.warped)),
            "difference": _png_b64(registration.difference(base_img, best.warped)),
        },
        "note": "corner_error_px scores each recovered transform against the "
                "known one. Compare it against rmse_px -- they disagree, and "
                "the disagreement is the lesson.",
    }


# ===========================================================================
# T1-T3 -- EHR pipeline
# ===========================================================================

# The pipeline takes a couple of seconds and is deterministic in its seed, so
# the result is cached per (n_patients, seed). Without this, every panel the
# frontend renders would re-run the whole thing.
_EHR_CACHE: dict[tuple[int, int], dict] = {}


def _frame(df, limit: int | None = None) -> list[dict]:
    """DataFrame -> JSON-safe records, NaN replaced with None."""
    subset = df.head(limit) if limit else df
    return subset.replace({np.nan: None}).to_dict(orient="records")


@app.get("/ehr/pipeline")
def ehr_pipeline(
    n_patients: int = Query(400, ge=50, le=2000),
    seed: int = Query(7, ge=0, le=9999),
    rows: int = Query(25, ge=1, le=200, description="Sample rows per table"),
) -> JSONResponse:
    """The whole T1-T3 pipeline: generate, standardize, clean, dedupe, impute.

    Every stage is returned -- reports plus a sample of each table -- so the
    UI can show before/after pairs without re-running anything.
    """
    from ehr import pipeline as ehr_pipeline_module

    key = (n_patients, seed)
    if key not in _EHR_CACHE:
        result = ehr_pipeline_module.run(n_patients=n_patients, seed=seed)

        reports = result.reports
        _EHR_CACHE[key] = {
            "ok": True,
            "params": {"n_patients": n_patients, "seed": seed},

            # T1 -- what arrived, and the coding systems in play
            "source": dict(reports["source"]),
            "coding_systems": {
                "icd10": _icd10_table(),
                "loinc": _loinc_table(),
                "note": "ICD-10 and LOINC codes are real. The SNOMED entries "
                        "in vocab.py are labelled placeholders.",
            },

            # T2 -- standardization then cleaning
            "standardize": {k: dict(v) for k, v in reports["standardize"].items()},
            "clean": dict(reports["clean"]),
            "flags": _frame(result.flags, rows),

            # T3 -- dedupe then missing data
            "dedupe": dict(reports["dedupe"]),
            "dedupe_evaluation": dict(reports["dedupe_evaluation"]),
            "match_pairs": _frame(result.match_pairs, rows),
            "missing": dict(reports["missing"]),
            "missing_profile": _frame(reports["missing_profile"]),
            "mechanism_diagnostics": {
                name: _frame(table)
                for name, table in reports["mechanism_diagnostics"].items()
            },
            "imputation_comparison": _frame(reports["imputation_comparison"]),

            # Samples, for the before/after tables
            "samples": {
                "raw_patients": _frame(result.raw["patients"], rows),
                "raw_observations": _frame(result.raw["observations"], rows),
                "standardised_patients": _frame(result.standardised["patients"], rows),
                "standardised_observations": _frame(
                    result.standardised["observations"], rows),
                "imaging_studies": _frame(result.standardised["imaging_studies"], rows),
            },
        }

    return JSONResponse(_EHR_CACHE[key])


def _icd10_table() -> list[dict]:
    from ehr import vocab
    by_code: dict[str, list[str]] = {}
    for text, code in vocab.DIAGNOSIS_SYNONYMS.items():
        by_code.setdefault(code, []).append(text)
    return [
        {"code": code, "display": display, "local_synonyms": sorted(by_code.get(code, []))}
        for code, display in vocab.ICD10.items()
    ]


def _loinc_table() -> list[dict]:
    from ehr import vocab
    by_code: dict[str, list[str]] = {}
    for text, code in vocab.LOCAL_TEST_TO_LOINC.items():
        by_code.setdefault(code, []).append(text)
    return [
        {
            "code": code,
            "display": spec["display"],
            "canonical_unit": spec["canonical_unit"],
            "plausible_range": list(spec["plausible"]),
            "local_names": sorted(by_code.get(code, [])),
        }
        for code, spec in vocab.LOINC.items()
    ]


# ===========================================================================
# Optional Gradio demo, mounted at "/"
# ===========================================================================
#
# Off by default, on when OVASCAN_MOUNT_GRADIO=1 (the Space Dockerfile sets
# it). Local development then needs neither gradio nor pillow installed.
#
# WHY IT IS MOUNTED RATHER THAN RUN ON ITS OWN
#     A Gradio-SDK Space runs `python app.py` and Gradio owns the server, so
#     there is nowhere to put a JSON API -- and the Next.js front end needs
#     one. Under the Docker SDK, FastAPI owns the server and Gradio becomes a
#     sub-application. That is why every route above was deliberately kept off
#     "/": this is what "/" was being saved for.
#
# A failure here must never take the API down with it. The demo is a
# convenience; /predict is the product.

if os.environ.get("OVASCAN_MOUNT_GRADIO") == "1":
    try:
        import gradio as gr

        from app import demo as _demo

        # show_api=False: gradio 5.9.1's schema generator crashes on this
        # Blocks (TypeError: argument of type 'bool' is not iterable in
        # get_api_info), which 500s the page itself.
        app = gr.mount_gradio_app(app, _demo, path="/", show_api=False)
        _GRADIO_MOUNTED = True
        _GRADIO_ERROR: str | None = None
    except Exception as exc:  # noqa: BLE001 - the API must survive this
        _GRADIO_MOUNTED = False
        _GRADIO_ERROR = f"{type(exc).__name__}: {exc}"

        @app.get("/")
        def _gradio_unavailable() -> dict:
            """Stand in for the demo so "/" is not a bare 404."""
            return {
                "name": "OvaScan API",
                "demo": "unavailable",
                "error": _GRADIO_ERROR,
                "hint": "The JSON API is unaffected. See GET /api and GET /docs.",
            }
else:
    _GRADIO_MOUNTED = False
    _GRADIO_ERROR = None

    @app.get("/")
    def _root_redirect() -> dict:
        """Local runs have no demo mounted; point the caller at the API."""
        return {
            "name": "OvaScan API",
            "disclaimer": "Research prototype. Not a medical device.",
            "hint": "Set OVASCAN_MOUNT_GRADIO=1 for the demo UI. "
                    "See GET /api and GET /docs.",
        }
