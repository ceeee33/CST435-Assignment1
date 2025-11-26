FROM python:3.11-slim

# OS deps (OpenCV, rembg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt

# Upgrade pip version
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy REST app sources
COPY . /app/

ENV PYTHONUNBUFFERED=1