---
title: OvaScan API
emoji: 🩺
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# OvaScan — ovarian cyst segmentation

**Research prototype. Not a medical device. Not for clinical use.**

This Space runs the full FastAPI service, with the Gradio demo mounted at `/`.

| Path | What |
|---|---|
| `/` | Gradio demo — upload a scan, see the mask and measurements |
| `/docs` | OpenAPI browser, every endpoint runnable from the page |
| `/health` | liveness, and which checkpoint is loaded |
| `/predict` | segmentation + morphometry (T4, T5) |
| `/image/enhance` | CLAHE, speckle filters, quality metrics (T4) |
| `/image/compress` | rate-distortion and measurement drift (T4) |
| `/image/register` | SIFT → Lowe ratio → RANSAC (T5) |
| `/image/register-and-segment` | register, segment both frames, compare |
| `/image/register/demo` | scored against a known transform |
| `/image/phantom` | synthetic ultrasound frame, no dataset needed |

Built for 21CSE428T Healthcare Analytics. Source and per-topic writeups:
<https://github.com/DrKingSchultz69/Ovarian_Cyst_Detection>

## Notes

- **Docker SDK, not Gradio.** The Gradio SDK owns the server and cannot expose a
  JSON API alongside it, which the web front end needs. Docker is a free,
  standard Space SDK; an earlier comment in this repo claiming otherwise was
  wrong.
- **Cold start.** The free tier sleeps after inactivity; the first request then
  takes ~40 s while torch loads. Hit `/health` once before a demo.
- **One worker.** The model is loaded per process and holds ~500 MB resident, so
  requests queue rather than fail.
- **CORS.** `OVASCAN_ALLOWED_ORIGINS` defaults to `*`. Set it to the deployed
  front end's origin under Settings → Variables.
