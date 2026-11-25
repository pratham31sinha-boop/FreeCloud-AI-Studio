FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libcairo2-dev \
    pkg-config \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-worker.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-worker.txt python-dotenv

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY github/ ./github/

RUN mkdir -p data/uploads data/artifacts data/outputs data/characters data/cache

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
