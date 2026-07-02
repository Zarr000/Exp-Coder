# Expera AI Local Deployment Dockerfile
# For GPU-enabled local inference

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    libaio7 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source
COPY src /app/src
COPY scripts /app/scripts
COPY checkpoints /app/checkpoints
COPY tokenizer /app/tokenizer

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    sentencepiece \
    aiohttp \
    fastapi \
    uvicorn \
    pydantic

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python3", "-m", "uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]