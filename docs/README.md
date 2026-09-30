# Syllabus coverage — 21CSE428T Healthcare Analytics

How OvaScan covers tutorial topics T1–T6, with a file and a page for each.

> **Scope.** This project covers **Unit-2 (T4–T6)** as application features.
> Unit-1 (T1–T3) is handled as a *dataset* exercise — selecting a valid dataset,
> standardising it, and removing redundancy — and lives in `data_prep/` as two
> audit scripts rather than as pages in the app. See [Unit-1](#unit-1--the-dataset-audits).

---

## The map

| | Topic | Backend | Frontend | Doc |
|---|---|---|---|---|
| **T4** | Enhancement, restoration, segmentation, compression | [`enhance.py`](../backend/enhance.py), [`inference.py`](../backend/inference.py), [`compress.py`](../backend/compress.py) | `/imaging`, `/` | [T4](T4-image-processing.md) |
| **T5** | SIFT, RANSAC, CNN | [`registration.py`](../backend/registration.py), [`inference.py`](../backend/inference.py) | `/registration`, `/` | [T5](T5-sift-ransac-cnn.md) |
| **T6** | Visualization | [`components/Charts.tsx`](../frontend/src/components/Charts.tsx) | every page | [T6](T6-visualization.md) |

## Unit-1 — the dataset audits

T1–T3 are answered against the **real MMOTU dataset**, not a synthetic cohort:

| Topic | Evidence | Script |
|---|---|---|
| **T1** selecting a valid dataset | 1469 images, 8 classes, official patient-split | `data_prep/mmotu_audit.py` |
| **T2** standardization, cleaning | 859 distinct image sizes; class ids → 5 ICD-10 codes; masks uniformly `[0,1]`; 0 corrupt files | `mmotu_audit.py`, `mmotu_image_audit.py` |
| **T3** redundancy, missing data | 1469 byte-identical duplicate masks (33.3%); **44 of 469 val images leak into train**; 0 empty masks | `mmotu_image_audit.py` |

```bash
python data_prep/mmotu_audit.py --zip "archive.zip"
```

```bash
python data_prep/mmotu_image_audit.py --zip "archive.zip"
```

The synthetic EHR cohort that previously backed T1–T3 has been removed: the tutorials
are about *this dataset*, and the real one turned out to have better material in it.

---

## Running it

Backend:

```bash
cd ovascan/backend && python -m venv .venv && .venv/Scripts/activate
```

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
```

```bash
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd ovascan/frontend && npm install && npm run dev
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local` first. It is
inlined at **build** time, so changing it later needs a rebuild, not just a restart.

### Without the model weights

`torch` and `ultralytics` are ~2.5 GB and only `/predict` needs them. Everything for
T1–T3 and the T4/T5 analysis runs without either:

```bash
pip install "numpy<2" pandas opencv-python-headless scikit-image scipy scikit-learn fastapi "uvicorn[standard]" python-multipart
```

`main.py` catches the missing import, `/health` reports
`segmentation_available: false`, and `/predict` returns 503 while every other endpoint
works. The three analysis pages need no checkpoint and no MMOTU download —
[`phantom.py`](../backend/phantom.py) generates a synthetic ultrasound frame instead.

Each module also runs standalone, no server involved:

```bash
python data_prep/mmotu_image_audit.py --zip "archive.zip"
```

```bash
python -m phantom
```

---

## What is real and what is synthetic

Being specific about this matters more than usual, because two of the three data
sources here are fabricated.

| | Status |
|---|---|
| MMOTU ultrasound images | **Real** public dataset (OTU_2d), used for training |
| YOLOv11n-seg weights | **Real**, trained on MMOTU — `weights/ovascan_burnin1.pt` |
| ICD-10 and LOINC codes | **Real** and verifiable |
| SNOMED CT entries | **Placeholders**, labelled as such in `vocab.py` |
| Phantom ultrasound frames | **Synthetic**, and not a wave simulation |

T1–T3 are measured against the real dataset. The defects found there were not planted
by us, which is what makes them worth reporting.

---

## Findings worth defending in a viva

Each is measured, reproducible from a seed, and documented where it was found.

1. **Blocking turns record linkage from intractable to trivial** — 87,990 candidate
   pairs become 22 comparisons, at F1 1.000 on the tuning seed and 0.974 held out.
   ([T3](T3-redundant-and-missing-data.md))
2. **The three missingness mechanisms leave three distinct diagnostic signatures** —
   MAR correlates with the variable that drives it (−0.426), MNAR leaks through a
   proxy (0.421), MCAR correlates with nothing (max 0.071). ([T3](T3-redundant-and-missing-data.md))
3. **A homography has the best reprojection RMSE and the worst true error**, by 5×.
   Overfitting made visible, and the reason RMSE can never choose a model.
   ([T5](T5-sift-ransac-cnn.md))
4. **SSIM is non-monotonic in JPEG quality on speckle images** while PSNR is not —
   evidence that SSIM's assumptions do not hold for ultrasound. ([T4](T4-image-processing.md))
5. **Histogram equalisation raises measured sharpness while making the image worse**,
   because on ultrasound there is little to sharpen except noise. ([T4](T4-image-processing.md))
6. **Fidelity degrades far faster than the measurement does** — SSIM 0.855 at a lesion
   area drift of −0.7% — but only for an easy target, and the doc says why that does
   not generalise. ([T4](T4-image-processing.md))

---

**Research prototype. Not a medical device. Not for clinical use.**
