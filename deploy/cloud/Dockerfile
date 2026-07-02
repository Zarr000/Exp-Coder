# Expera AI Cloud Deployment Dockerfile
# Optimized for cloud GPU instances (AWS g5, GCP a2, Azure NC series)

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    wget \
    vim \
    htop \
    numactl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create app user
RUN useradd -m -s /bin/bash appuser || true
RUN mkdir -p /app && chown -R appuser:appuser /app

# Set work directory
WORKDIR /app

# Copy requirements
COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY src /app/src

# Create necessary directories
RUN mkdir -p /app/checkpoints
RUN mkdir -p /app/tokenizer
RUN mkdir -p /app/cache
RUN mkdir -p /app/data
RUN mkdir -p /app/logs
RUN mkdir -p /app/outputs

# Create non-root user ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server with multiple workers for production
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]