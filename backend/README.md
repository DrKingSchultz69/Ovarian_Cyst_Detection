# OvaScan API — Phase 1

Ovarian cyst segmentation service. YOLOv11n-seg via Ultralytics, wrapped in FastAPI.

**Research prototype. Not a medical device. Not for clinical use.**

---

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on mac/linux
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then put the trained checkpoint at `weights/ovascan_burnin1.pt`, or point
`OVASCAN_WEIGHTS` at it. The service will not import without weights.

## Verify inference before starting the server

```bash
python smoke_test.py test.jpg
```

All five checks must pass. Look at `smoke_out.png` to confirm the mask sits on
the lesion — a green run of checks with a mask in the wrong place still means
the model is wrong.

## Run

```bash
uvicorn main:app --reload --port 8000
```

- `http://localhost:8000/docs` — upload an image in the browser, see the result.
  This page is a usable fallback demo if the frontend is not ready.
- `http://localhost:8000/health` — liveness, plus the loaded weights filename.

```bash
curl -F "file=@test.jpg" -F "conf=0.25" http://localhost:8000/predict
```

---

## Response contract

Frozen. Phase 2 builds against this shape, so change it here and update the
frontend mock in the same commit.

```json
{
  "ok": true,
  "conf": 0.25,
  "count": 2,
  "latency_ms": 340,
  "units": "pixels",
  "image_size": { "width": 640, "height": 480 },
  "annotated_png": "<base64 PNG>",
  "note": "All distances and areas are in image pixels...",
  "lesions": [
    {
      "id": 1,
      "conf": 0.871,
      "bbox": [120, 88, 302, 240],
      "max_diameter_px": 142.3,
      "area_px": 9821,
      "circularity": 0.812,
      "solidity": 0.944,
      "echo_mean": 71.2,
      "echo_sd": 18.5
    }
  ]
}
```

`POST /predict` with `debug=true` adds a `debug` object holding `raw_png`,
`burnin_mask_png` and `inpainted_png` — the three preprocessing panels.

On failure: `{"ok": false, "error": "ValueError: ...", "count": 0, "lesions": []}`
with HTTP 422. The error string is safe to render in the UI.

### Status codes

| Code | Meaning |
|------|---------|
| 200 | Inference ran. `count` may still be 0 — that is a valid negative result, not an error. |
| 413 | File over 10 MB. |
| 415 | Content type is not an accepted image format. |
| 422 | Decodable request, but inference failed. Read `error`. |

---

## Notes

- **Units are pixels.** mm conversion needs the scanner scale bar or DICOM
  `PixelSpacing`, neither of which this prototype reads.
- **Model loads once at import**, with a warm-up pass. Do not move the `YOLO()`
  call into a request handler.
- **`count: 0` is a result, not a failure.** The UI must render "no lesion
  detected" rather than an error state.

## Deployment gotchas

- `opencv-python` instead of `opencv-python-headless` imports fine locally and
  crashes on a slim server image with a missing `libGL.so.1`.
- Free-tier hosts sleep after ~15 min idle; the first request after that takes
  ~40 s. Hit `/health` before any live demo.
- `OVASCAN_ALLOWED_ORIGINS` defaults to `*`. Set it to the real frontend origin
  before the service is public.
