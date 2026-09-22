---
title: OvaScan API
emoji: 🩺
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# OvaScan v0.1

Ovarian cyst segmentation in ultrasound images (YOLOv11n-seg).

**Research prototype. Not a medical device. Not for clinical use.**

- `/` — Gradio UI
- `POST /predict` — multipart `file`, `conf`, `debug`; returns JSON
- `GET /health` — liveness + loaded weights
- `/docs` — interactive API docs
