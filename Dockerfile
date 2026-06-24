# Backend image for ForensicAI (Flask API served via gunicorn).
FROM python:3.12-slim

# Don't write .pyc files; flush stdout/stderr immediately for container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries required by OpenCV (opencv-python) and video I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
# Install CPU-only torch/torchvision from the PyTorch CPU index first. This
# avoids the ~3GB of CUDA/nvidia wheels the default (GPU) build would pull into
# a container that has no GPU. The subsequent -r install then sees torch as
# already satisfied.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

# Copy the backend source. app.py imports `backend.api.*` / `backend.models.*`,
# which resolve as namespace packages with /app on the path.
COPY backend/ ./backend/

# The Flask app reads PORT (default 5000); gunicorn binds the same port.
ENV PORT=5000
EXPOSE 5000

# Required at runtime (the app refuses to start without them):
#   -e API_KEY=...           shared secret for the X-API-Key header
#   -e CORS_ORIGINS=...      comma-separated allowed origins
# The app object lives at backend/api/app.py as `app`.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 300 backend.api.app:app"]
