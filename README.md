# OvaScan v0.2

Ovarian cyst segmentation in ultrasound, plus the EHR and image-analysis pipelines
around it. Built for **21CSE428T Healthcare Analytics**, covering tutorial topics
T1–T6.

**Research prototype. Not a medical device. Not for clinical use.**

---

## What is here

```
backend/
  inference.py      YOLOv11n-seg segmentation + burn-in restoration   T4, T5
  enhance.py        CLAHE, speckle reduction, sharpening + metrics    T4
  compress.py       rate-distortion and measurement-drift analysis    T4
  registration.py   SIFT -> ratio test -> RANSAC -> warp              T5
  phantom.py        synthetic ultrasound fixture (no dataset needed)
  evaluate.py       IoU/Dice against real MMOTU masks                 T5
  ehr/              generate -> standardize -> clean -> dedupe -> impute  T1-T3
  main.py           FastAPI transport layer over all of the above

frontend/           Next.js 16, four pages, one per topic group       T6
data_prep/          MMOTU masks -> YOLO labels, plus a data-quality audit  T1-T3
docs/               one writeup per syllabus topic
```

Every analysis module is free of web-framework imports and runs standalone in a REPL —
the same rule `inference.py` already followed, because debugging segmentation or a
pandas pipeline through HTTP is far slower than debugging it directly.

## Syllabus coverage

| Topic | Where |
|---|---|
| T1 Understanding EHR | `ehr/vocab.py`, `ehr/generate.py` · `/ehr#t1` |
| T2 Standardization, cleaning | `ehr/standardize.py`, `ehr/clean.py` · `/ehr#t2` |
| T3 Redundancy, missing data | `ehr/dedupe.py`, `ehr/missing.py` · `/ehr#t3` |
| T4 Enhance, restore, segment, compress | `enhance.py`, `inference.py`, `compress.py` · `/imaging`, `/` |
| T5 SIFT, RANSAC, CNN | `registration.py`, `inference.py` · `/registration`, `/` |
| T6 Visualization | `frontend/src/components/Charts.tsx` · all pages |

Full mapping, findings and per-topic writeups: **[`docs/README.md`](docs/README.md)**.

The two halves describe **one cohort** — every synthetic patient owns one MMOTU
ultrasound, joined on `imaging_studies.scan_stem`.

## Quick start

```bash
cd backend && python -m venv .venv && .venv/Scripts/activate
```

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
```

```bash
uvicorn main:app --reload --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local`. It is inlined
at **build** time — changing it later needs a rebuild, not a restart.

### Without the 2.5 GB of torch

Only `/predict` needs torch, ultralytics and the checkpoint. Skip them and everything
else still works — `/health` reports `segmentation_available: false`, `/predict`
returns 503, and the EHR, enhancement, compression and registration endpoints run
normally against the built-in phantom.

```bash
pip install "numpy<2" pandas opencv-python-headless scikit-image scipy scikit-learn fastapi "uvicorn[standard]" python-multipart
```

### Run the pipelines directly

```bash
python -m ehr.pipeline
```

```bash
python -m phantom
```

```bash
python smoke_test.py test.jpg
```

## Data provenance

| | Status |
|---|---|
| MMOTU images, YOLO weights | real |
| ICD-10, LOINC codes | real |
| SNOMED CT entries | **placeholders**, labelled in `vocab.py` |
| Patient cohort | **entirely synthetic** — no real patient data |
| Phantom frames | synthetic; not a wave simulation |

The cohort is synthetic by design: T2 and T3 measure how well defects are *repaired*,
and that needs the uncorrupted original, which no real extract ships. See
[`docs/T1`](docs/T1-understanding-ehr.md) for why Synthea and public clinical CSVs were
considered and rejected.

## Deployment

[`DEPLOY.md`](DEPLOY.md) — Hugging Face Spaces for the backend, Vercel for the
frontend. Note that `requirements.txt` and the `Dockerfile` now also carry pandas,
scikit-learn, scikit-image and scipy, and the image copies `ehr/` and the four new
modules.

## Units

**Every distance and area is in image pixels.** Converting to mm needs the scanner
scale bar or DICOM `PixelSpacing`, neither of which this prototype reads.
