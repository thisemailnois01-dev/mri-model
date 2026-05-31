# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps for OpenCV ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ───────────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (layer cache) ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app ──────────────────────────────────────────────────────────────────
COPY . .

# ── Run ───────────────────────────────────────────────────────────────────────
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
