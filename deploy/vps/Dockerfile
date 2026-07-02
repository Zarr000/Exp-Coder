# Expera AI VPS Deployment Dockerfile
# Optimized for VPS with dedicated GPU

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9

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
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create app user
RUN useradd -m -s /bin/bash appuser || true
RUN mkdir -p /app && chown -R appuser:appuser /app

# Set work directory
WORKDIR /app

# Copy requirements first for caching
COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY src /app/src
COPY src/api /app/src/api
COPY src/runtime /app/src/runtime
COPY src/memory /app/src/memory
COPY src/rag /app/src/rag
COPY src/agents /app/src/agents
COPY src/multimodal /app/src/multimodal
COPY src/tools /app/src/tools
COPY src/training /app/src/training

# Copy configs
COPY configs /app/configs

# Copy model files (should be mounted)
RUN mkdir -p /app/checkpoints
RUN mkdir -p /app/tokenizer

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]