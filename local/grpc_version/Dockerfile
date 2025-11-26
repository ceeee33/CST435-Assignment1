# Use a Python 3.9 slim base image for a smaller footprint
FROM python:3.9-slim

# Build-time service selector (python by default). Set to 'uploader' to enable Node.js layer
ARG SERVICE=python

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies
# libgl1-mesa-glx and libglib2.0-0 are often needed for OpenCV and other graphical libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# Optionally install Node.js when building the uploader image
RUN if [ "$SERVICE" = "uploader" ]; then \
            apt-get update && apt-get install -y curl ca-certificates gnupg && \
            curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
            apt-get install -y nodejs && \
            node --version && npm --version && \
            rm -rf /var/lib/apt/lists/*; \
        fi


# Copy the requirements file and install Python dependencies
COPY requirements.txt .

# Upgrade pip version
RUN pip install --upgrade pip

# Install Python libraries (use requirements; optional wheels step removed for simplicity)
RUN pip install --no-cache-dir -r requirements.txt

# Optionally install JS deps for uploader (copy root package.json)
COPY package.json ./services/package.json
COPY package-lock.json* ./services/
RUN if [ "$SERVICE" = "uploader" ]; then \
            cd services && (npm ci --only=production || npm install --only=production); \
        fi

# Copy the rest of your application code
COPY . /app/

# Generate gRPC stubs for pipeline.proto
RUN python generate_protos.py

# Expose the gRPC port
EXPOSE 50030 50040 50041 50042 50043

ENV PYTHONUNBUFFERED=1

# Run the gRPC server when the container starts
# Default command (overridden by docker-compose.yml for microservices)
CMD ["python", "services/processor_service.py"]
