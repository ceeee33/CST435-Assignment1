FROM python:3.11-slim

# Build-time service selector (python by default). Set to 'uploader' to enable Node.js layer
ARG SERVICE=python

WORKDIR /app

# OS deps (OpenCV, rembg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

 # Optionally install Node.js when building the uploader image
RUN if [ "$SERVICE" = "uploader" ]; then \
            apt-get update && apt-get install -y curl ca-certificates gnupg && \
            curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
            apt-get install -y nodejs && \
            node --version && npm --version && \
            rm -rf /var/lib/apt/lists/*; \
        fi

# Install Python deps
COPY requirements.txt /app/requirements.txt

# Upgrade pip version
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Optionally install JS deps for uploader
COPY package.json ./services/package.json
RUN if [ "$SERVICE" = "uploader" ]; then \
            cd services && (npm ci --only=production || npm install --only=production); \
        fi

# Copy REST app sources
COPY . /app/

ENV PYTHONUNBUFFERED=1