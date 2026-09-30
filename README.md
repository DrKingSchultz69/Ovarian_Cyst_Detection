# OvaScan v0.2

Ovarian cyst segmentation in ultrasound, plus the enhancement, compression and
registration analysis around it. Built for **21CSE428T Healthcare Analytics**, covering tutorial topics
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
  main.py           FastAPI transport layer over all of the above

frontend/           Next.js 16, three pages                           T6
data_prep/          MMOTU -> YOLO labels, plus two dataset audits      T1-T3
docs/               one writeup per syllabus topic
```

Every analysis module is free of web-framework imports and runs standalone in a REPL —
the same rule `inference.py` already followed, because debugging segmentation or a
pandas pipeline through HTTP is far slower than debugging it directly.

## Syllabus coverage

| Topic | Where |
|---|---|
| T1–T3 Dataset selection, standardization, redundancy | `data_prep/mmotu_audit.py`, `data_prep/mmotu_image_audit.py` |
| T4 Enhance, restore, segment, compress | `enhance.py`, `inference.py`, `compress.py` · `/imaging`, `/` |
| T5 SIFT, RANSAC, CNN | `registration.py`, `inference.py` · `/registration`, `/` |
| T6 Visualization | `frontend/src/components/Charts.tsx` · all pages |

Full mapping, findings and per-topic writeups: **[`docs/README.md`](docs/README.md)**.

T1–T3 are a **dataset** exercise run against the real MMOTU archive; T4–T6 are
application features. The synthetic EHR cohort that previously backed T1–T3 has been
removed.

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
returns 503, and the enhancement, compression and registration endpoints run
normally against the built-in phantom.

```bash
pip install "numpy<2.3" opencv-python-headless scikit-image scipy fastapi "uvicorn[standard]" python-multipart
```

### Run the pipelines directly

```bash
python data_prep/mmotu_image_audit.py --zip "archive.zip"
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
| Phantom frames | synthetic; not a wave simulation |

T1–T3 are measured against the real dataset — see [`docs/README.md`](docs/README.md).

## Deployment

[`DEPLOY.md`](DEPLOY.md) — Hugging Face Spaces for the backend, Vercel for the
frontend. Note that `requirements.txt` and the `Dockerfile` now also carry pandas,
scikit-image and scipy, and the image copies the analysis modules.

## Units

**Every distance and area is in image pixels.** Converting to mm needs the scanner
scale bar or DICOM `PixelSpacing`, neither of which this prototype reads.
