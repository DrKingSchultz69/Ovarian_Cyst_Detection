# Hugging Face Spaces (Docker SDK). Spaces routes traffic to port 7860.
FROM python:3.11-slim

# opencv-python-headless still needs libgl/libglib at runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs as uid 1000; anything Ultralytics writes must be under $HOME.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# CPU-only torch first, or pip resolves the ~2.5 GB CUDA build via ultralytics.
RUN pip install --no-cache-dir --user \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user inference.py main.py ./
# T4/T5 analysis modules and the synthetic-ultrasound fixture.
COPY --chown=user enhance.py compress.py registration.py phantom.py ./
COPY --chown=user weights/ ./weights/

EXPOSE 7860

# One worker: the model is loaded per process and holds ~500 MB resident.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
