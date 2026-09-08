# Expera AI Dockerfile
# Local inference or VPS deployment

FROM python:3.11-slim

# Set environment
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose ports
# API server
EXPOSE 8000

# Build for different configurations:

# Base (CPU inference)
# docker build -t expera-ai:base .

# GPU (CUDA inference)
# docker build -t expera-ai:gpu --build-arg USE_GPU=1 .

# VPS (Production)
# docker build -t expera-ai:vps --build-arg DEPLOY=vps .

# Entrypoint
ENTRYPOINT ["python", "-m", "src.server.app"]