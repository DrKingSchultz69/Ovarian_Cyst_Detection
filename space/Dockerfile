# Hugging Face Space, Docker SDK. Spaces routes traffic to app_port (7860).
#
# Docker is a free, standard Space SDK. It is used instead of the Gradio SDK
# because Gradio owns the server process and cannot serve a JSON API next to
# the UI -- and the Next.js front end needs that API. Here FastAPI owns the
# server and the Gradio demo is mounted at "/".
FROM python:3.11-slim

# opencv-python-headless still needs libgl/libglib at runtime on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs as uid 1000; anything written at runtime must live under $HOME.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    PYTHONUNBUFFERED=1 \
    OVASCAN_MOUNT_GRADIO=1

WORKDIR $HOME/app

# CPU-only torch FIRST, or pip resolves ultralytics' torch dependency from
# PyPI and pulls the ~2.5 GB CUDA build into the image.
RUN pip install --no-cache-dir --user \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user *.py ./
COPY --chown=user weights/ ./weights/

EXPOSE 7860

# One worker: the model is loaded per process and holds ~500 MB resident.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
